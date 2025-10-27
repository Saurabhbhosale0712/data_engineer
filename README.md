# data_engineer

Nice — below is a **complete, production-ready Databricks notebook** (single Python file you can paste into a Databricks Python notebook).
It uses a **hybrid LLM + regex** approach, processes data **partition-wise** (no `.collect()`), requires no hard-coded tokens (uses `dbutils.secrets`), and **MERGE**-es results into your Delta table. Every block has clear comments explaining what it does.

> ⚠️ Before running:
>
> 1. Create a Databricks Secret Scope (example: `llm_scope`) and store your model token under key `databricks_token`.
> 2. Set `SRC_TABLE`, `TARGET_TABLE`, and `LLM_ENDPOINT` to your values.
> 3. Tune `BATCH_SIZE` for token / latency tradeoffs.

---

```python
# Databricks notebook: LLM + regex hybrid extractor (id-wise)
# Purpose: extract structured fields (applicant_name, cibil_score, loan_amount, loan_purpose, dpd, emi)
# from free-text recommendation notes and save into Delta table (merge/upsert).
#
# Key features:
# - Light cleaning of notes
# - Partitioned processing using mapInPandas (avoids collect())
# - Calls LLM in batches; LLM must return a JSON array mapping to input ids
# - Robust parsing + fallback regex extraction when LLM fails / returns malformed output
# - Safe secret handling (dbutils.secrets.get)
# - Merge into Delta to avoid overwrite

# -------------------------
# Step 0: Imports & config
# -------------------------
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType
)
import pandas as pd
import re, json, time, requests
from typing import Iterator

# ---------- USER CONFIG ----------
SRC_TABLE = "bfl_std_lake.risk_sme_secured.recommendation_notes"  # source table with LOAN_APPLICATION__C and RECOMMENDATION_COMMENTS__C
TARGET_TABLE = "bfl_std_lake.risk_sme_secured.customer_notes_insights"  # target delta table
LOAN_ID_COL = "LOAN_APPLICATION__C"   # id column name in source
NOTE_COL = "RECOMMENDATION_COMMENTS__C"  # notes column name in source

# Databricks model serving endpoint (replace with your workspace endpoint)
LLM_ENDPOINT = "https://adb-xxxxxx.azuredatabricks.net/serving-endpoints/databricks-claude-sonnet-4-5/invocations"

# Batch size for LLM calls (tune smaller if notes are long)
BATCH_SIZE = 8

# Get token securely from Databricks secret scope (DO NOT hardcode)
TOKEN = dbutils.secrets.get(scope="llm_scope", key="databricks_token")
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
# ---------------------------------

# ---------------------------
# Step 1: Lightweight cleaner
# ---------------------------
# Clean notes but keep financial tokens and punctuation useful for regex
def clean_text_keep_fin(t: str) -> str:
    if t is None:
        return ""
    s = re.sub(r"\s+", " ", str(t)).strip()
    # keep ASCII, digits, basic punctuation and common financial symbols
    s = re.sub(r"[^\x00-\x7F]", " ", s)
    s = re.sub(r"[^\w\s\.,:%₹$\/\-\(\)]+", "", s)
    return s.strip()

# Expose as simple Pandas-safe function (we'll call inside mapInPandas)
def _clean_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).apply(clean_text_keep_fin)

# ------------------------------------------------------
# Step 2: Prompt template & robust response parser + retry
# ------------------------------------------------------
PROMPT_HEADER = """
You are a precise data-extraction assistant. For each input note produce a JSON object with fields:
- id (string): the input id provided
- applicant_name (string) or empty string
- cibil_score (integer) or null
- loan_amount (number, INR) or null
- loan_purpose (string) or empty string
- dpd (string e.g., '0','30','60+') or null
- emi (number) or null
- raw_insight (short string summary) or empty string

Return a single JSON array of objects in the SAME ORDER as the inputs.
Return ONLY valid JSON (no explanation, no bullet points).
Example:
[{"id":"L1","applicant_name":"John Doe","cibil_score":720,"loan_amount":1000000,"loan_purpose":"home","dpd":null,"emi":15000,"raw_insight":"..."}, ...]
Now extract for the following notes:
"""

# Helper: parse likely response JSON strings returned by various model servers
def parse_possible_response_json(resp_json):
    # resp_json: python object from requests.json() OR a string
    if isinstance(resp_json, dict):
        # common keys used by various servers
        for key in ("predictions", "outputs", "output", "results", "choices"):
            if key in resp_json:
                val = resp_json[key]
                # normalize list -> combined string
                if isinstance(val, list):
                    parts = []
                    for it in val:
                        if isinstance(it, dict):
                            for sub in ("content","text","output"):
                                if sub in it:
                                    parts.append(it[sub])
                                    break
                            else:
                                parts.append(json.dumps(it))
                        elif isinstance(it, str):
                            parts.append(it)
                        else:
                            parts.append(json.dumps(it))
                    return "\n\n".join(parts)
                elif isinstance(val, str):
                    return val
        # fallback keys
        for key in ("message","result"):
            if key in resp_json and isinstance(resp_json[key], str):
                return resp_json[key]
        return None
    elif isinstance(resp_json, list):
        # convert list to string
        return json.dumps(resp_json)
    else:
        # not a dict/list
        return None

# Robust LLM caller with simple exponential backoff
def call_llm(prompt: str, n_retries=3, timeout=60):
    payload = {"messages":[{"role":"user","content":prompt}]}
    last_exc = None
    for attempt in range(1, n_retries+1):
        try:
            resp = requests.post(LLM_ENDPOINT, headers=HEADERS, json=payload, timeout=timeout)
            if resp.status_code != 200:
                last_exc = Exception(f"Status {resp.status_code}: {resp.text[:200]}")
                time.sleep(2 ** attempt)
                continue
            # try to parse JSON body from server
            try:
                j = resp.json()
            except Exception:
                j = None
            text = parse_possible_response_json(j) if j is not None else None
            if text is None:
                # try raw text fallback
                text = resp.text
            return text
        except Exception as e:
            last_exc = e
            time.sleep(2 ** attempt)
    # if all retries fail, raise last exception
    raise last_exc

# ------------------------------------------
# Step 3: Regex fallback extractors (deterministic)
# ------------------------------------------
# These are fast, deterministic extractors for numeric/pattern fields
cibil_re = re.compile(r"\b([3-9][0-9]{2})\b")   # 300-999 capture
money_re = re.compile(r"(?:₹|Rs\.?|INR)?\s*([0-9]+(?:[,\.][0-9]{2,3})*)(?:\s*(lakh|lac|lakhs|lakhs|crore|cr|k|m))?", flags=re.I)
emi_re = re.compile(r"\bEMI[:\s]*([0-9,]+)\b", re.I)
dpd_re = re.compile(r"\b(\d{1,3})\s*(?:dpd|days past due|days)\b", re.I)
name_hint_re = re.compile(r"(?:Applicant|Customer|Borrower|Name)[:\s\-|]*([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+){0,2})")

def fallback_extract_single(note_text: str) -> dict:
    # Returns a dict with keys applicant_name, cibil_score, loan_amount, loan_purpose, dpd, emi
    out = {"applicant_name": "", "cibil_score": None, "loan_amount": None, "loan_purpose": "", "dpd": None, "emi": None}
    if not note_text:
        return out
    # CIBIL
    m = cibil_re.search(note_text)
    if m:
        val = int(m.group(1))
        if 300 <= val <= 900:
            out["cibil_score"] = val
    # EMI
    m = emi_re.search(note_text)
    if m:
        try:
            out["emi"] = float(m.group(1).replace(",", ""))
        except:
            out["emi"] = None
    # DPD
    m = dpd_re.search(note_text)
    if m:
        out["dpd"] = m.group(1)
    # Name hint
    m = name_hint_re.search(note_text)
    if m:
        out["applicant_name"] = m.group(1).strip()
    # Money / loan amount
    m = money_re.search(note_text)
    if m:
        raw = m.group(1).replace(",", "")
        scale = m.group(2)
        try:
            val = float(raw)
            if scale:
                s = scale.lower()
                if "lakh" in s or "lac" in s:
                    val *= 100000
                elif "crore" in s or "cr" in s:
                    val *= 10000000
                elif s == "k":
                    val *= 1000
                elif s == "m":
                    val *= 1000000
            out["loan_amount"] = val
        except:
            out["loan_amount"] = None
    # Loan purpose simple heuristic - look for "for <purpose>"
    m = re.search(r"\bfor\s+([a-zA-Z ]{3,60}?)(?:[.,;()]|$)", note_text, re.I)
    if m:
        out["loan_purpose"] = m.group(1).strip()
    return out

# ------------------------------------------------------------
# Step 4: mapInPandas processor - processes partition Pandas dfs
# ------------------------------------------------------------
# This function will be executed per partition and should yield DataFrames
def process_partition(pdf_iter: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """
    pdf_iter: iterator of pandas DataFrames (each chunk = a partition slice)
    We will:
      - Clean notes
      - Send notes in small batches to LLM
      - Parse JSON result; if parse fails -> use regex fallback for that batch
      - Yield a pandas DataFrame with result rows (one row per input id)
    """
    for pdf in pdf_iter:
        # Ensure relevant columns exist
        if LOAN_ID_COL not in pdf.columns or NOTE_COL not in pdf.columns:
            # nothing to do; yield empty
            yield pd.DataFrame(columns=[
                "loan_application_id","applicant_name","cibil_score","loan_amount",
                "loan_purpose","dpd","emi","raw_insight","extraction_source"
            ])
            continue

        # Clean notes
        pdf["cleaned_note"] = _clean_series(pdf[NOTE_COL])

        results = []  # accumulate dicts to return
        # iterate in batches (list of rows)
        n = len(pdf)
        for i in range(0, n, BATCH_SIZE):
            batch = pdf.iloc[i:i+BATCH_SIZE]
            # build prompt for the batch
            prompt = PROMPT_HEADER
            for _, row in batch.iterrows():
                # include id and the cleaned note (triple-quote to preserve newlines)
                prompt += f'\nID: "{row[LOAN_ID_COL]}"\nNote: """{row["cleaned_note"]}"""\n'

            # call LLM and try to parse JSON
            try:
                model_text = call_llm(prompt)
            except Exception as e:
                # LLM call failed for entire batch -> fallback for each note
                for _, row in batch.iterrows():
                    fb = fallback_extract_single(row["cleaned_note"])
                    results.append({
                        "loan_application_id": str(row[LOAN_ID_COL]),
                        "applicant_name": fb["applicant_name"],
                        "cibil_score": fb["cibil_score"],
                        "loan_amount": fb["loan_amount"],
                        "loan_purpose": fb["loan_purpose"],
                        "dpd": fb["dpd"],
                        "emi": fb["emi"],
                        "raw_insight": "",
                        "extraction_source": "fallback"
                    })
                # continue to next batch
                continue

            # try to parse model_text as JSON
            parsed = None
            try:
                parsed = json.loads(model_text)
                # if a dict with list nested, try to extract first list
                if isinstance(parsed, dict):
                    for v in parsed.values():
                        if isinstance(v, list):
                            parsed = v
                            break
            except Exception:
                # try to heuristically extract JSON substring
                start = model_text.find('[')
                end = model_text.rfind(']') + 1
                try:
                    if start != -1 and end != -1:
                        parsed = json.loads(model_text[start:end])
                except Exception:
                    parsed = None

            # If parsed is not a list -> fallback per note
            if parsed is None or not isinstance(parsed, list):
                for _, row in batch.iterrows():
                    fb = fallback_extract_single(row["cleaned_note"])
                    results.append({
                        "loan_application_id": str(row[LOAN_ID_COL]),
                        "applicant_name": fb["applicant_name"],
                        "cibil_score": fb["cibil_score"],
                        "loan_amount": fb["loan_amount"],
                        "loan_purpose": fb["loan_purpose"],
                        "dpd": fb["dpd"],
                        "emi": fb["emi"],
                        "raw_insight": "",
                        "extraction_source": "fallback"
                    })
                continue

            # parsed is a list; map each parsed object to the input rows (by id or by order)
            for idx, (_, row) in enumerate(batch.iterrows()):
                loan_id = str(row[LOAN_ID_COL])
                mapped = None
                # primary: index match if lengths align and parsed[index].get("id") matches or is empty
                if idx < len(parsed) and isinstance(parsed[idx], dict):
                    # if parsed has id and it matches this loan_id, accept it
                    pid = str(parsed[idx].get("id","")) if isinstance(parsed[idx].get("id",""), (str,int)) else ""
                    if pid == "" or pid == loan_id:
                        mapped = parsed[idx]
                # secondary: find parsed item with matching id
                if mapped is None:
                    for p in parsed:
                        if isinstance(p, dict) and str(p.get("id","")) == loan_id:
                            mapped = p
                            break
                # tertiary: fallback to index-th parsed object if dict
                if mapped is None and idx < len(parsed) and isinstance(parsed[idx], dict):
                    mapped = parsed[idx]

                if mapped is None:
                    # cannot match -> fallback regex for this single note
                    fb = fallback_extract_single(row["cleaned_note"])
                    results.append({
                        "loan_application_id": loan_id,
                        "applicant_name": fb["applicant_name"],
                        "cibil_score": fb["cibil_score"],
                        "loan_amount": fb["loan_amount"],
                        "loan_purpose": fb["loan_purpose"],
                        "dpd": fb["dpd"],
                        "emi": fb["emi"],
                        "raw_insight": "",
                        "extraction_source": "fallback"
                    })
                    continue

                # sanitize mapped values carefully
                def safe_int(x):
                    try:
                        if x is None: return None
                        return int(x)
                    except:
                        try:
                            return int(float(str(x).replace(",","")))
                        except:
                            return None
                def safe_float(x):
                    try:
                        if x is None: return None
                        return float(str(x).replace(",",""))
                    except:
                        return None

                results.append({
                    "loan_application_id": loan_id,
                    "applicant_name": (mapped.get("applicant_name") or "").strip(),
                    "cibil_score": safe_int(mapped.get("cibil_score")),
                    "loan_amount": safe_float(mapped.get("loan_amount")),
                    "loan_purpose": (mapped.get("loan_purpose") or "").strip(),
                    "dpd": (mapped.get("dpd") or None),
                    "emi": safe_float(mapped.get("emi")),
                    "raw_insight": (mapped.get("raw_insight") or "").strip(),
                    "extraction_source": "llm"
                })

        # yield results for this partition as pandas DataFrame
        if len(results) == 0:
            # ensure correct schema even if no rows
            yield pd.DataFrame(columns=[
                "loan_application_id","applicant_name","cibil_score","loan_amount",
                "loan_purpose","dpd","emi","raw_insight","extraction_source"
            ])
        else:
            yield pd.DataFrame(results)

# -------------------------------
# Step 5: Run extraction over table
# -------------------------------
# Read source table (only required columns)
df_src = spark.table(SRC_TABLE).select(LOAN_ID_COL, NOTE_COL)

# Use mapInPandas to process partitions safely (each partition processed by process_partition)
output_schema = StructType([
    StructField("loan_application_id", StringType(), True),
    StructField("applicant_name", StringType(), True),
    StructField("cibil_score", IntegerType(), True),
    StructField("loan_amount", DoubleType(), True),
    StructField("loan_purpose", StringType(), True),
    StructField("dpd", StringType(), True),
    StructField("emi", DoubleType(), True),
    StructField("raw_insight", StringType(), True),
    StructField("extraction_source", StringType(), True)
])

# This runs code per partition; result_df is a Spark DataFrame with extracted fields
result_df = df_src.mapInPandas(process_partition, schema=output_schema)

# Optional: peek at a sample
display(result_df.limit(20))

# --------------------------------------
# Step 6: Persist results into Delta (MERGE upsert)
# --------------------------------------
# Create/merge into TARGET_TABLE on loan_application_id
from delta.tables import DeltaTable

if spark._jsparkSession.catalog().tableExists(TARGET_TABLE):
    # write results to a temp view for SQL MERGE
    temp_view = "tmp_llm_extracted"
    result_df.createOrReplaceTempView(temp_view)

    merge_sql = f"""
    MERGE INTO {TARGET_TABLE} T
    USING {temp_view} S
      ON T.loan_application_id = S.loan_application_id
    WHEN MATCHED THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
    """
    spark.sql(merge_sql)
    print(f"Merged results into existing table {TARGET_TABLE}")
else:
    # create new table (overwrite if needed)
    result_df.write.format("delta").mode("overwrite").saveAsTable(TARGET_TABLE)
    print(f"Created table {TARGET_TABLE} and wrote results")

# -----------------------------
# Step 7: Quick verification sample
# -----------------------------
print("Sample from target table:")
display(spark.sql(f"SELECT * FROM {TARGET_TABLE} ORDER BY loan_application_id LIMIT 50"))
```

---

## Short explanation of what this does (TL;DR)

1. Reads your source table with recommendation notes.
2. Cleans notes (keeps ₹, numbers, punctuation helpful for extraction).
3. Processes data partition-by-partition (scales for large tables).
4. For each partition it:

   * Sends small batches to the LLM with a **strict JSON-only** instruction.
   * Tries to parse the returned JSON and align results `id` → input id.
   * If the LLM fails to return valid JSON or mapping, it runs **deterministic regex-based extraction** for that note.
5. Stores results with a field `extraction_source` indicating `llm` or `fallback`.
6. MERGE-s results into a Delta table to avoid overwriting previous runs.

---

If you want now I can:

* Produce a **pure LLM-only** variant (no regex fallbacks) for quick experiments, or
* Add a **confidence** field by asking the LLM to return a per-field confidence (and implement threshold routing), or
* Convert this into a **Databricks Job** with logging/metrics.

Which one should I give you next?
