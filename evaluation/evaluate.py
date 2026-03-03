"""
Mean Recall@K Evaluation
Computes how well the recommendation system performs vs. the labeled training set.

Usage:
    set GEMINI_API_KEY=your_key
    python evaluation/evaluate.py
"""

import os
import sys
import json
import csv
import re
import time
import numpy as np
import requests
from collections import defaultdict

# ─── paths ───────────────────────────────────────
HERE     = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(HERE)
TRAIN_CSV = os.path.join(HERE, "train_set.csv")
CSV_PATH  = os.path.join(ROOT, "shl_assessments_v2.csv")
if not os.path.exists(CSV_PATH):
    CSV_PATH = os.path.join(ROOT, "shl_assessments.csv")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBED_MODEL  = "text-embedding-004"
FLASH_MODEL  = "gemini-1.5-flash"
BASE_API     = "https://generativelanguage.googleapis.com/v1beta"
HEADERS_JSON = {"Content-Type": "application/json"}


# ─── load catalog once ───────────────────────────
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


# ─── gemini helpers ──────────────────────────────
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


def recommend(query_text, rows, matrix, valid_idx, top_n=10):
    enriched = expand_query(query_text)
    q_vec = get_embedding(enriched)
    sims  = cosine_sim(q_vec, matrix)
    top_i = np.argsort(sims)[::-1][:top_n]
    urls  = []
    for i in top_i:
        row = rows[valid_idx[i]]
        url = row.get("url", "").strip().rstrip("/").lower()
        urls.append(url)
    return urls


# ─── recall@k ────────────────────────────────────
def recall_at_k(predicted: list, relevant: set, k: int = 10) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for u in predicted[:k] if u.rstrip("/").lower() in relevant)
    return hits / len(relevant)


# ─── main eval ───────────────────────────────────
def main():
    print(f"Loading catalog: {CSV_PATH}")
    rows, matrix, valid_idx = load_catalog()
    print(f"  {len(rows)} rows, {len(valid_idx)} with embeddings\n")

    # Load train set → group by query
    ground_truth = defaultdict(set)
    with open(TRAIN_CSV, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("Query", "").strip()
            u = row.get("Assessment_url", "").strip().rstrip("/").lower()
            if q and u:
                ground_truth[q].add(u)

    queries = list(ground_truth.keys())
    print(f"Evaluating on {len(queries)} unique queries...\n")

    recall_scores = []
    for qi, query in enumerate(queries):
        relevant = ground_truth[query]
        print(f"[{qi+1}/{len(queries)}] Query: {query[:80]}...")
        try:
            predicted = recommend(query, rows, matrix, valid_idx, top_n=10)
            score = recall_at_k(predicted, relevant, k=10)
            recall_scores.append(score)
            print(f"  Relevant: {len(relevant)}  Predicted hits: {sum(1 for u in predicted if u in relevant)}  Recall@10: {score:.3f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            recall_scores.append(0.0)
        time.sleep(0.5)  # rate limit

    mean_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
    print(f"\n{'='*60}")
    print(f"Mean Recall@10: {mean_recall:.4f}")
    print(f"{'='*60}")
    return mean_recall


if __name__ == "__main__":
    main()
