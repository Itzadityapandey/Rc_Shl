"""
Local development server for the Python API.
Runs Flask on port 5000 so that Next.js (next.config.ts rewrites) can proxy /api/* to it.

Usage:
    set GEMINI_API_KEY=your-key
    python dev_server.py
"""

from flask import Flask, request, Response, jsonify
import json, os, csv, re, time, sys
import numpy as np
import requests as http_requests

app = Flask(__name__)

# ── Inline the same logic from api/recommend.py ──────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBED_MODEL    = "text-embedding-004"
FLASH_MODEL    = "gemini-1.5-flash"
BASE_API       = "https://generativelanguage.googleapis.com/v1beta"
HEADERS_JSON   = {"Content-Type": "application/json"}

HERE     = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "shl_assessments_v2.csv")
if not os.path.exists(CSV_PATH):
    CSV_PATH = os.path.join(HERE, "shl_assessments.csv")

_df_rows = _emb_matrix = _valid_indices = None


def _load():
    global _df_rows, _emb_matrix, _valid_indices
    if _df_rows is not None:
        return
    rows, embeddings, valid_idx = [], [], []
    with open(CSV_PATH, newline="", encoding="utf-8", errors="replace") as f:
        for i, row in enumerate(csv.DictReader(f)):
            rows.append(row)
            try:
                emb = json.loads(row.get("embedding", "[]") or "[]")
                if isinstance(emb, list) and len(emb) > 0:
                    embeddings.append(emb)
                    valid_idx.append(i)
            except Exception:
                pass
    _df_rows = rows
    _emb_matrix = np.array(embeddings, dtype=np.float32) if embeddings else np.array([])
    _valid_indices = valid_idx
    print(f"[catalog] {len(rows)} rows, {len(valid_idx)} embedded.")


def _get_embedding(text):
    url = f"{BASE_API}/models/{EMBED_MODEL}:embedContent?key={GEMINI_API_KEY}"
    payload = {"model": f"models/{EMBED_MODEL}", "content": {"parts": [{"text": text[:8000]}]}}
    r = http_requests.post(url, json=payload, headers=HEADERS_JSON, timeout=15)
    r.raise_for_status()
    return np.array(r.json()["embedding"]["values"], dtype=np.float32)


def _expand_query(text):
    url = f"{BASE_API}/models/{FLASH_MODEL}:generateContent?key={GEMINI_API_KEY}"
    prompt = (
        "Extract key skills, competencies, and assessment needs from this job query "
        "into a concise paragraph for semantic HR assessment search (max 150 words):\n\n"
        f"{text[:3000]}\n\nEnriched text:"
    )
    try:
        r = http_requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, headers=HEADERS_JSON, timeout=20)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return text


def _cosine(q, M):
    q = q / (np.linalg.norm(q) + 1e-10)
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-10)
    return M @ q


def _parse_types(raw):
    if not raw or str(raw).strip() in ("N/A", "nan", ""):
        return []
    mapping = {"personality":"P","behavior":"P","ability":"A","aptitude":"A",
               "cognitive":"A","reasoning":"A","knowledge":"K","skill":"K",
               "biodata":"B","simulation":"S","situational":"S","competency":"C"}
    types = set()
    for kw, code in mapping.items():
        if kw in str(raw).lower():
            types.add(code)
    return list(types) if types else [str(raw)[:3].upper()]


def _parse_dur(raw):
    if not raw or str(raw).strip() in ("N/A", "nan", ""):
        return 0
    m = re.search(r"(\d+)", str(raw))
    return int(m.group(1)) if m else 0


def _recommend(query_text, top_n=10):
    _load()
    enriched = _expand_query(query_text)
    q_vec = _get_embedding(enriched)
    sims  = _cosine(q_vec, _emb_matrix)
    ranked = np.argsort(sims)[::-1]

    results, p_cnt, o_cnt = [], 0, 0
    p_slots = max(2, top_n // 3)
    o_slots = top_n - p_slots
    for ri in ranked:
        if len(results) >= top_n:
            break
        row = _df_rows[_valid_indices[ri]]
        types = _parse_types(row.get("test_type", ""))
        if "P" in types and p_cnt < p_slots:
            results.append((ri, row, sims[ri]))
            p_cnt += 1
        elif "P" not in types and o_cnt < o_slots:
            results.append((ri, row, sims[ri]))
            o_cnt += 1
    selected = {r[0] for r in results}
    for ri in ranked:
        if len(results) >= top_n:
            break
        if ri not in selected:
            results.append((ri, _df_rows[_valid_indices[ri]], sims[ri]))

    out = []
    for _, row, _ in results[:top_n]:
        out.append({
            "url": row.get("url", ""),
            "adaptive_support": str(row.get("adaptive_support","No")).strip().capitalize(),
            "description": str(row.get("description","")).strip()[:300],
            "duration": _parse_dur(row.get("duration","")),
            "remote_support": str(row.get("remote_support","No")).strip().capitalize(),
            "test_type": _parse_types(row.get("test_type","")),
        })
    return out


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


@app.route("/api/recommend", methods=["POST", "OPTIONS"])
def recommend():
    if request.method == "OPTIONS":
        return "", 204, {"Access-Control-Allow-Origin": "*",
                         "Access-Control-Allow-Methods": "POST, OPTIONS",
                         "Access-Control-Allow-Headers": "Content-Type"}
    data = request.get_json(silent=True) or {}
    q  = (data.get("query") or data.get("job_description") or "").strip()
    ju = (data.get("job_url") or "").strip()
    if not q and not ju:
        return jsonify({"error": "Provide 'query' or 'job_url'"}), 400
    if ju and not q:
        try:
            r = http_requests.get(ju, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            r.raise_for_status()
            q = re.sub(r"<[^>]+>", " ", r.text)
            q = re.sub(r"\s+", " ", q).strip()[:5000]
        except Exception as e:
            return jsonify({"error": f"Failed to scrape URL: {e}"}), 400
    try:
        recs = _recommend(q)
        return Response(json.dumps({"recommended_assessments": recs}),
                        mimetype="application/json",
                        headers={"Access-Control-Allow-Origin": "*"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"Starting dev server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
