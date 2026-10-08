"""
preprocessing.py
----------------
Handles loading and cleaning the job dataset.

Steps (visible for viva explanation):
  1. Load CSV
  2. Handle missing values
  3. Remove duplicates
  4. Text cleaning (lowercase, strip special chars)
  5. Combine relevant columns into a single 'combined_text' field
"""

import re
import pandas as pd


# ─────────────────────────────────────────────────────────────
# Column name mapping
# The lukebarousse/data_jobs dataset uses these column names.
# If you swap the CSV, update the mapping here.
# ─────────────────────────────────────────────────────────────
COLUMN_MAP = {
    "title"       : ["job_title", "title", "Title", "Job Title"],
    "company"     : ["company_name", "company", "Company"],
    "location"    : ["job_location", "location", "Location"],
    "description" : ["job_description", "description", "Description"],
    "skills"      : ["job_skills", "skills", "Skills", "Required Skills"],
    "type"        : ["job_schedule_type", "employment_type", "job_type", "Type"],
    "category"    : ["job_title_short", "category", "Category"],
}


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column name that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def clean_text(text: str) -> str:
    """
    Basic text cleaning:
      - Convert to lowercase
      - Remove special characters (keep letters, digits, spaces, commas)
      - Collapse multiple spaces
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\+\#]", " ", text)  # keep + # . for C++, C#, etc.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_preprocess(csv_path: str, sample_size: int = 3000) -> pd.DataFrame:
    """
    Load the CSV, clean it, and return a processed DataFrame.

    Parameters
    ----------
    csv_path   : path to the jobs CSV file
    sample_size: max rows to keep (keeps app snappy for demos)

    Returns
    -------
    df : cleaned DataFrame with standardised column names:
         title, company, location, description, skills, category, combined_text
    """

    # ── 1. Load ──────────────────────────────────────────────
    print(f"[Preprocessing] Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Raw rows   : {len(df)}")
    print(f"  Raw columns: {list(df.columns)}")

    # ── 2. Map columns to standard names ─────────────────────
    rename = {}
    for standard_name, candidates in COLUMN_MAP.items():
        found = _find_col(df, candidates)
        if found:
            rename[found] = standard_name

    df.rename(columns=rename, inplace=True)

    # Ensure all standard columns exist (fill with empty string if missing)
    for col in ["title", "company", "location", "description", "skills", "category", "type"]:
        if col not in df.columns:
            df[col] = ""

    # ── 3. Handle missing values ──────────────────────────────
    # Pandas 3.x requires type-aware fill: string cols → "" , numeric → 0
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].fillna("")
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(0)

    # ── 4. Remove duplicates ──────────────────────────────────
    before_dedup = len(df)
    df.drop_duplicates(subset=["title", "company", "location"], inplace=True)
    print(f"  After dedup: {len(df)} rows (removed {before_dedup - len(df)} duplicates)")

    # ── 5. Text cleaning ──────────────────────────────────────
    for col in ["title", "company", "location", "description", "skills", "category"]:
        df[col] = df[col].apply(clean_text)

    # Drop rows where title is completely empty after cleaning
    df = df[df["title"].str.strip() != ""]

    # ── 6. Sample (for performance) ───────────────────────────
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"  Sampled to : {len(df)} rows")

    # ── 7. Combine relevant text into one field ───────────────
    # This combined_text is what TF-IDF will operate on.
    df["combined_text"] = (
        df["title"]       + " " +
        df["category"]    + " " +
        df["skills"]      + " " +
        df["description"].str[:500]   # limit description length for TF-IDF
    )
    df["combined_text"] = df["combined_text"].apply(clean_text)

    print(f"[Preprocessing] Done. Final shape: {df.shape}")
    return df
