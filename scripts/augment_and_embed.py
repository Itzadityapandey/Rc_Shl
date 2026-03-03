"""
One-time offline script to:
1. Merge the company-provided Excel (Gen_AI Dataset.xlsx) with shl_assessments.csv
2. Pre-compute Gemini embeddings for any rows missing them
3. Save the merged result as shl_assessments_v2.csv

Run this ONCE before deploying. The API never calls embeddings at request time.
Usage: python scripts/augment_and_embed.py
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import requests

# ────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBEDDING_MODEL = "text-embedding-004"
EMBED_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{EMBEDDING_MODEL}:embedContent?key={GEMINI_API_KEY}"
)
HEADERS = {"Content-Type": "application/json"}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRAPED_CSV = os.path.join(BASE_DIR, "shl_assessments.csv")
EXCEL_FILE  = os.path.join(BASE_DIR, "Gen_AI Dataset.xlsx")
OUTPUT_CSV  = os.path.join(BASE_DIR, "shl_assessments_v2.csv")


# ────────────────────────────────────────────
# EMBEDDING HELPER
# ────────────────────────────────────────────
def get_embedding(text: str, retries: int = 3) -> list:
    """Call Gemini text-embedding-004 and return the vector."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY env variable not set.")
    payload = {
        "model": f"models/{EMBEDDING_MODEL}",
        "content": {"parts": [{"text": text[:8000]}]},  # API limit
    }
    for attempt in range(retries):
        try:
            r = requests.post(EMBED_URL, json=payload, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return r.json()["embedding"]["values"]
        except Exception as e:
            print(f"  [embed attempt {attempt+1}] Error: {e}")
            time.sleep(2 ** attempt)
    return []


# ────────────────────────────────────────────
# STEP 1 – Load scraped CSV
# ────────────────────────────────────────────
print(f"Loading scraped CSV: {SCRAPED_CSV}")
df_scraped = pd.read_csv(SCRAPED_CSV)
print(f"  Scraped rows: {len(df_scraped)}  columns: {list(df_scraped.columns)}")

# Normalise URL column for merging
df_scraped["url"] = df_scraped["url"].str.strip().str.rstrip("/").str.lower()


# ────────────────────────────────────────────
# STEP 2 – Load Excel (Train + Test sets only have queries/urls, not catalog data)
#           The Excel gives us labelled Q→URL pairs, NOT the assessment catalog.
#           We extract unique assessment URLs from Train-Set to make sure they
#           are all present in our catalog.
# ────────────────────────────────────────────
print(f"\nLoading Excel: {EXCEL_FILE}")
train_df = pd.read_excel(EXCEL_FILE, sheet_name="Train-Set")
test_df  = pd.read_excel(EXCEL_FILE, sheet_name="Test-Set")

print(f"  Train-Set shape: {train_df.shape}")
print(f"  Test-Set  shape: {test_df.shape}")

# Get all unique assessment URLs referenced in training labels
label_urls = set(
    train_df["Assessment_url"].dropna().str.strip().str.rstrip("/").str.lower()
)
print(f"  Unique assessment URLs in train labels: {len(label_urls)}")

# Find which label URLs are missing from scraped catalog
missing_urls = label_urls - set(df_scraped["url"])
print(f"  URLs missing from scraped CSV: {len(missing_urls)}")
if missing_urls:
    for u in sorted(missing_urls)[:20]:
        print(f"    {u}")


# ────────────────────────────────────────────
# STEP 3 – Create stub rows for missing URLs so they can be embedded
# ────────────────────────────────────────────
stub_rows = []
for url in missing_urls:
    # Extract a humanised name from the slug
    slug = url.split("/")[-1].replace("-", " ").replace("_", " ").title()
    stub_rows.append({
        "name": slug,
        "url": url,
        "description": slug,
        "duration": "N/A",
        "test_type": "Unknown",
        "remote_support": "Unknown",
        "adaptive_support": "Unknown",
        "embedding": "",
    })

if stub_rows:
    df_stubs = pd.DataFrame(stub_rows)
    df_all = pd.concat([df_scraped, df_stubs], ignore_index=True)
    print(f"\nAdded {len(stub_rows)} stub rows. Total catalog size: {len(df_all)}")
else:
    df_all = df_scraped.copy()
    print("\nNo stub rows needed. All label URLs are in the catalog.")

print(f"Final catalog size before dedup: {len(df_all)}")
df_all = df_all.drop_duplicates(subset=["url"], keep="first")
print(f"Final catalog size after dedup:  {len(df_all)}")


# ────────────────────────────────────────────
# STEP 4 – Compute embeddings for rows that are missing them
# ────────────────────────────────────────────
# Check which rows already have valid embeddings
def is_valid_embedding(val):
    if pd.isna(val) or val == "" or val is None:
        return False
    try:
        parsed = json.loads(val)
        return isinstance(parsed, list) and len(parsed) > 0
    except Exception:
        return False

df_all["has_embedding"] = df_all["embedding"].apply(is_valid_embedding)
needs_embedding = df_all[~df_all["has_embedding"]]
print(f"\nRows needing embeddings: {len(needs_embedding)}")

if len(needs_embedding) == 0:
    print("  All rows already have embeddings. Skipping embedding step.")
else:
    if not GEMINI_API_KEY:
        print("ERROR: Set GEMINI_API_KEY env var to compute embeddings.")
        sys.exit(1)

    for i, (idx, row) in enumerate(needs_embedding.iterrows()):
        text = f"{row['name']}. {row['description']}"
        print(f"  [{i+1}/{len(needs_embedding)}] Embedding: {row['name'][:60]}")
        emb = get_embedding(text)
        if emb:
            df_all.at[idx, "embedding"] = json.dumps(emb)
        else:
            print(f"    WARNING: Failed to embed row {idx}")
        # Rate limit: ~1500 RPM for free tier → ~25 RPS
        time.sleep(0.05)

df_all = df_all.drop(columns=["has_embedding"])


# ────────────────────────────────────────────
# STEP 5 – Save merged catalog
# ────────────────────────────────────────────
df_all.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print(f"\n✅ Saved {len(df_all)} assessments to: {OUTPUT_CSV}")

# Also save evaluation sets
train_out = os.path.join(BASE_DIR, "evaluation", "train_set.csv")
test_out  = os.path.join(BASE_DIR, "evaluation", "test_set.csv")
os.makedirs(os.path.dirname(train_out), exist_ok=True)
train_df.to_csv(train_out, index=False, encoding="utf-8")
test_df.to_csv(test_out, index=False, encoding="utf-8")
print(f"   Saved train set → {train_out}")
print(f"   Saved test  set → {test_out}")
