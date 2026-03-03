"""
Precompute embeddings for all rows in shl_assessments.csv that are missing them.
This is an alternative to augment_and_embed.py — use this if you only have the
scraped CSV and want to fill in any rows that failed during initial crawling.

Usage:
    set GEMINI_API_KEY=your-key
    python scripts/precompute_embeddings.py
"""

import os
import sys
import json
import csv
import time
import requests

HERE     = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(HERE)
# Try v2 first, then fall back to original
CSV_IN   = os.path.join(ROOT, "shl_assessments_v2.csv")
if not os.path.exists(CSV_IN):
    CSV_IN = os.path.join(ROOT, "shl_assessments.csv")
CSV_OUT  = CSV_IN  # overwrite in-place

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBED_MODEL  = "text-embedding-004"
BASE_API     = "https://generativelanguage.googleapis.com/v1beta"


def get_embedding(text: str, retries: int = 3) -> list:
    url = f"{BASE_API}/models/{EMBED_MODEL}:embedContent?key={GEMINI_API_KEY}"
    payload = {
        "model": f"models/{EMBED_MODEL}",
        "content": {"parts": [{"text": text[:8000]}]}
    }
    for attempt in range(retries):
        try:
            r = requests.post(url, json=payload,
                              headers={"Content-Type": "application/json"}, timeout=30)
            r.raise_for_status()
            return r.json()["embedding"]["values"]
        except Exception as e:
            print(f"  [attempt {attempt+1}] Error: {e}")
            time.sleep(2 ** attempt)
    return []


def has_embedding(val) -> bool:
    if not val or str(val).strip() in ("", "[]", "nan"):
        return False
    try:
        parsed = json.loads(val)
        return isinstance(parsed, list) and len(parsed) > 0
    except Exception:
        return False


def main():
    if not GEMINI_API_KEY:
        print("ERROR: Set GEMINI_API_KEY env var first.")
        sys.exit(1)

    print(f"Loading: {CSV_IN}")
    rows = []
    with open(CSV_IN, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    print(f"  Total rows: {len(rows)}")
    missing = [i for i, r in enumerate(rows) if not has_embedding(r.get("embedding", ""))]
    print(f"  Missing embeddings: {len(missing)}")

    if not missing:
        print("✅ All rows already have embeddings.")
        return

    for count, idx in enumerate(missing):
        row = rows[idx]
        name = row.get("name", "")
        desc = row.get("description", "")
        text = f"{name}. {desc}".strip()
        print(f"[{count+1}/{len(missing)}] Embedding: {name[:60]}")
        emb = get_embedding(text)
        if emb:
            rows[idx]["embedding"] = json.dumps(emb)
        else:
            print(f"  WARNING: Failed for row {idx}")
        # Avoid rate limit (free tier ~1500 RPM)
        time.sleep(0.05)

    # Save back
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Saved {len(rows)} rows → {CSV_OUT}")
    filled = sum(1 for r in rows if has_embedding(r.get("embedding", "")))
    print(f"   Rows with embeddings: {filled}/{len(rows)}")


if __name__ == "__main__":
    main()
