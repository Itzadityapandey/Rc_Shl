"""
POST /api/recommend
Body: { "query": "...", "job_url": "..." }  (at least one required)

Optimized Vercel Serverless Function:
- CSV + embeddings loaded ONCE into global memory (warm invocations reuse cache)
- Zero Selenium, zero runtime scraping
- Gemini Flash for LLM query-expansion (1 fast call)
- Balanced recommendations: mix of Personality (P) and Skill/Cognitive (A, B, K, S) types
"""

from http.server import BaseHTTPRequestHandler
import json
import os
import csv
import io
import re
import time
import numpy as np
import requests

# ─────────────────────────────────────────────────────────────
# GLOBAL In-Memory Cache (persists across warm invocations)
# ─────────────────────────────────────────────────────────────
_df_rows       = None   # list of dicts (assessment catalog)
_emb_matrix    = None   # np.ndarray shape (N, D)
_valid_indices = None   # list[int] → row indices with valid embeddings

GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
EMBED_MODEL       = "text-embedding-004"
FLASH_MODEL       = "gemini-1.5-flash"
BASE_API          = "https://generativelanguage.googleapis.com/v1beta"
HEADERS_JSON      = {"Content-Type": "application/json"}

# Path to CSV (relative to the repo root, Vercel copies all files)
_HERE    = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_HERE, "..", "shl_assessments_v2.csv")
if not os.path.exists(CSV_PATH):
    CSV_PATH = os.path.join(_HERE, "..", "shl_assessments.csv")


# ─────────────────────────────────────────────────────────────
# LOAD + CACHE
# ─────────────────────────────────────────────────────────────
def _load_catalog():
    global _df_rows, _emb_matrix, _valid_indices
    if _df_rows is not None:
        return  # already loaded

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

    _df_rows       = rows
    _emb_matrix    = np.array(embeddings, dtype=np.float32) if embeddings else np.array([])
    _valid_indices = valid_idx
    print(f"[catalog] Loaded {len(rows)} rows, {len(valid_idx)} with embeddings.")


# ─────────────────────────────────────────────────────────────
# GEMINI HELPERS
# ─────────────────────────────────────────────────────────────
def _get_embedding(text: str) -> np.ndarray:
    url = f"{BASE_API}/models/{EMBED_MODEL}:embedContent?key={GEMINI_API_KEY}"
    payload = {
        "model": f"models/{EMBED_MODEL}",
        "content": {"parts": [{"text": text[:8000]}]},
    }
    r = requests.post(url, json=payload, headers=HEADERS_JSON, timeout=15)
    r.raise_for_status()
    return np.array(r.json()["embedding"]["values"], dtype=np.float32)


def _expand_query(raw_query: str) -> str:
    """Use Gemini Flash to extract a clean, enriched query for better embedding recall."""
    url = f"{BASE_API}/models/{FLASH_MODEL}:generateContent?key={GEMINI_API_KEY}"
    prompt = (
        "You are an expert HR consultant. Given the job description or query below, "
        "extract and summarize the KEY SKILLS, COMPETENCIES, and ASSESSMENT NEEDS into a "
        "concise paragraph (max 150 words) optimized for semantic search against an HR "
        "assessment catalog. Include both technical and behavioral aspects.\n\n"
        f"Query/JD:\n{raw_query[:3000]}\n\n"
        "Enriched search text:"
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        r = requests.post(url, json=payload, headers=HEADERS_JSON, timeout=20)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"[expand_query] fallback to raw query. Error: {e}")
        return raw_query


def _scrape_url_fast(url: str) -> str:
    """Light HTTP scrape (no Selenium) for job URLs."""
    try:
        hdrs = {"User-Agent": "Mozilla/5.0 (compatible; SHLBot/1.0)"}
        r = requests.get(url, headers=hdrs, timeout=10)
        r.raise_for_status()
        text = re.sub(r"<[^>]+>", " ", r.text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:5000]
    except Exception as e:
        return ""


# ─────────────────────────────────────────────────────────────
# COSINE SIMILARITY
# ─────────────────────────────────────────────────────────────
def _cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    m_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return m_norm @ q_norm


# ─────────────────────────────────────────────────────────────
# BALANCED RANKING
# ─────────────────────────────────────────────────────────────
def _parse_test_types(raw) -> list:
    if not raw or raw == "nan":
        return []
    raw = str(raw)
    # Typical values: "Personality & Behavior" → "P", "Ability & Aptitude" → "A", etc
    mapping = {
        "personality": "P", "behavior": "P", "behaviour": "P",
        "ability": "A", "aptitude": "A",
        "biodata": "B",
        "knowledge": "K", "skill": "K",
        "simulation": "S", "situational": "S",
        "cognitive": "A", "reasoning": "A",
        "competency": "C",
    }
    types = set()
    lower = raw.lower()
    for kw, code in mapping.items():
        if kw in lower:
            types.add(code)
    return list(types) if types else [raw[:3].upper()]


def _balanced_top_n(sims: np.ndarray, valid_idx: list, rows: list, top_n: int = 10):
    """
    Return top_n indices with a balance between Personality (P) and Cognitive/Skill (A,K,B,S).
    Strategy: fill first half with highest-scored mixed types, then top overall for the rest.
    """
    ranked = np.argsort(sims)[::-1]  # descending similarity
    results = []
    p_count, other_count = 0, 0
    p_slots    = max(2, top_n // 3)      # at least 1/3 personality
    other_slots = top_n - p_slots

    # First pass: balanced
    for rank_i in ranked:
        if len(results) >= top_n:
            break
        row_idx = valid_idx[rank_i]
        row = rows[row_idx]
        types = _parse_test_types(row.get("test_type", ""))
        if "P" in types and p_count < p_slots:
            results.append((rank_i, row_idx, sims[rank_i]))
            p_count += 1
        elif "P" not in types and other_count < other_slots:
            results.append((rank_i, row_idx, sims[rank_i]))
            other_count += 1

    # Second pass: fill remaining slots greedily
    selected_rank_i = {r[0] for r in results}
    for rank_i in ranked:
        if len(results) >= top_n:
            break
        if rank_i not in selected_rank_i:
            row_idx = valid_idx[rank_i]
            results.append((rank_i, row_idx, sims[rank_i]))

    return results


# ─────────────────────────────────────────────────────────────
# DURATION PARSING
# ─────────────────────────────────────────────────────────────
def _parse_duration(raw) -> int:
    if not raw or str(raw).strip() in ("N/A", "nan", ""):
        return 0
    m = re.search(r"(\d+)", str(raw))
    return int(m.group(1)) if m else 0


# ─────────────────────────────────────────────────────────────
# MAIN HANDLER
# ─────────────────────────────────────────────────────────────
class handler(BaseHTTPRequestHandler):

    def do_OPTIONS(self):
        self._send_cors()

    def _send_cors(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _json_response(self, code: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(length) if length else b"{}"
            body = json.loads(raw_body.decode("utf-8"))
        except Exception:
            self._json_response(400, {"error": "Invalid JSON body"})
            return

        query_text = (body.get("query") or body.get("job_description") or "").strip()
        job_url    = (body.get("job_url") or "").strip()

        if not query_text and not job_url:
            self._json_response(400, {"error": "Provide 'query' or 'job_url' in request body"})
            return

        # Scrape URL if provided
        if job_url and not query_text:
            query_text = _scrape_url_fast(job_url)
            if not query_text:
                self._json_response(400, {"error": "Failed to scrape job URL. Provide query text instead."})
                return

        # Load catalog (cached after first call)
        _load_catalog()
        if _emb_matrix is None or len(_emb_matrix) == 0:
            self._json_response(500, {"error": "Assessment catalog not available"})
            return

        # LLM query expansion
        enriched = _expand_query(query_text)

        # Embed the enriched query
        try:
            q_vec = _get_embedding(enriched)
        except Exception as e:
            self._json_response(500, {"error": f"Embedding failed: {e}"})
            return

        # Cosine similarity
        sims = _cosine_similarity(q_vec, _emb_matrix)

        # Balanced top-10
        top_results = _balanced_top_n(sims, _valid_indices, _df_rows, top_n=10)

        # Build response
        recommendations = []
        for _, row_idx, sim_score in top_results:
            row = _df_rows[row_idx]
            test_types = _parse_test_types(row.get("test_type", ""))
            recommendations.append({
                "url": row.get("url", ""),
                "adaptive_support": str(row.get("adaptive_support", "No")).strip().capitalize(),
                "description": str(row.get("description", "")).strip()[:300],
                "duration": _parse_duration(row.get("duration", "")),
                "remote_support": str(row.get("remote_support", "No")).strip().capitalize(),
                "test_type": test_types,
            })

        if not recommendations:
            self._json_response(404, {"error": "No recommendations found"})
            return

        self._json_response(200, {"recommended_assessments": recommendations})
