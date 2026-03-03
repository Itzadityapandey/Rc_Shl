"""
Generate predictions.csv for the 9 unlabeled test queries.
Output format: query, assessment_url  (one row per recommendation per query)

Usage:
    set GEMINI_API_KEY=your_key
    python evaluation/generate_predictions.py
"""

import os
import sys
import json
import csv
import time
import numpy as np
import requests

HERE     = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(HERE)
TEST_CSV  = os.path.join(HERE, "test_set.csv")
CSV_PATH  = os.path.join(ROOT, "shl_assessments_v2.csv")
if not os.path.exists(CSV_PATH):
    CSV_PATH = os.path.join(ROOT, "shl_assessments.csv")
OUT_PATH  = os.path.join(HERE, "predictions.csv")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBED_MODEL  = "text-embedding-004"
FLASH_MODEL  = "gemini-1.5-flash"
BASE_API     = "https://generativelanguage.googleapis.com/v1beta"
HEADERS_JSON = {"Content-Type": "application/json"}


def load_catalog():
    rows, embeddings, valid_idx = [], [], []
    with open(CSV_PATH, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            rows.append(row)
            try:
                emb = json.loads(row.get("embedding", "[]") or "[]")
                if isinstance(emb, list) and len(emb) > 0:
                    embeddings.append(emb)
                    valid_idx.append(i)
            except Exception:
                pass
    matrix = np.array(embeddings, dtype=np.float32) if embeddings else np.array([])
    return rows, matrix, valid_idx


def get_embedding(text):
    url = f"{BASE_API}/models/{EMBED_MODEL}:embedContent?key={GEMINI_API_KEY}"
    payload = {"model": f"models/{EMBED_MODEL}", "content": {"parts": [{"text": text[:8000]}]}}
    r = requests.post(url, json=payload, headers=HEADERS_JSON, timeout=20)
    r.raise_for_status()
    return np.array(r.json()["embedding"]["values"], dtype=np.float32)


def expand_query(text):
    url = f"{BASE_API}/models/{FLASH_MODEL}:generateContent?key={GEMINI_API_KEY}"
    prompt = (
        "Extract key skills, competencies, and assessment needs from this job query "
        "into a concise paragraph for semantic HR assessment search (max 150 words):\n\n"
        f"{text[:3000]}\n\nEnriched text:"
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        r = requests.post(url, json=payload, headers=HEADERS_JSON, timeout=20)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return text


def cosine_sim(q_vec, matrix):
    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-10)
    m_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return m_norm @ q_norm


def parse_test_types(raw):
    if not raw or str(raw).strip() in ("N/A", "nan", ""):
        return []
    raw_str = str(raw).lower()
    mapping = {
        "personality": "P", "behavior": "P", "behaviour": "P",
        "ability": "A", "aptitude": "A", "cognitive": "A", "reasoning": "A",
        "biodata": "B",
        "knowledge": "K", "skill": "K",
        "simulation": "S", "situational": "S",
        "competency": "C",
    }
    types = set()
    for kw, code in mapping.items():
        if kw in raw_str:
            types.add(code)
    return list(types) if types else [str(raw)[:3].upper()]


def recommend(query_text, rows, matrix, valid_idx, top_n=10):
    enriched = expand_query(query_text)
    q_vec = get_embedding(enriched)
    sims  = cosine_sim(q_vec, matrix)
    ranked = np.argsort(sims)[::-1]

    results, p_count, other_count = [], 0, 0
    p_slots = max(2, top_n // 3)
    other_slots = top_n - p_slots

    for rank_i in ranked:
        if len(results) >= top_n:
            break
        row = rows[valid_idx[rank_i]]
        types = parse_test_types(row.get("test_type", ""))
        if "P" in types and p_count < p_slots:
            results.append(row.get("url", "").strip())
            p_count += 1
        elif "P" not in types and other_count < other_slots:
            results.append(row.get("url", "").strip())
            other_count += 1

    # Fill remaining greedily
    selected = set(results)
    for rank_i in ranked:
        if len(results) >= top_n:
            break
        url = rows[valid_idx[rank_i]].get("url", "").strip()
        if url not in selected:
            results.append(url)

    return results[:top_n]


def main():
    print(f"Loading catalog: {CSV_PATH}")
    rows, matrix, valid_idx = load_catalog()
    print(f"  {len(rows)} rows loaded\n")

    # Load test queries
    test_queries = []
    with open(TEST_CSV, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("Query", "").strip()
            if q:
                test_queries.append(q)

    print(f"Generating predictions for {len(test_queries)} test queries...\n")

    prediction_rows = []
    for qi, query in enumerate(test_queries):
        print(f"[{qi+1}/{len(test_queries)}] {query[:80]}...")
        try:
            preds = recommend(query, rows, matrix, valid_idx, top_n=10)
            for url in preds:
                prediction_rows.append({"query": query, "assessment_url": url})
            print(f"  → {len(preds)} recommendations")
        except Exception as e:
            print(f"  ERROR: {e}")
        time.sleep(1.0)  # avoid rate limiting

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "assessment_url"])
        writer.writeheader()
        writer.writerows(prediction_rows)

    print(f"\n✅ Saved {len(prediction_rows)} prediction rows → {OUT_PATH}")


if __name__ == "__main__":
    main()
