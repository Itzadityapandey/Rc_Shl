# SHL Assessment Recommender

An AI-powered web application that recommends SHL assessments based on job descriptions or natural language queries.

## 🔗 Submission URLs

| Resource | URL |
|---|---|
| **GitHub** | https://github.com/Itzadityapandey/Rc_Shl |
| **API Health** | `https://<your-vercel-app>.vercel.app/api/health` |
| **API Recommend** | `https://<your-vercel-app>.vercel.app/api/recommend` |
| **Frontend** | `https://<your-vercel-app>.vercel.app` |

> Replace `<your-vercel-app>` once deployed on Vercel.

---

## 🏗️ Architecture

```
User Query / JD / URL
        ↓
Next.js Frontend (React)
        ↓ POST /api/recommend
Python Serverless Function (Vercel)
        ↓
  1. Gemini Flash query expansion (LLM, 1 call)
  2. Gemini text-embedding-004 (embed query)
  3. Cosine similarity on 321+ pre-embedded assessments
  4. Balanced ranking (Personality + Cognitive mix)
        ↓
JSON response (≤10 assessments)
```

---

## 📋 API Specification

### `GET /api/health`
```json
{"status": "healthy"}
```

### `POST /api/recommend`
**Request:**
```json
{
  "query": "Looking for Java developer assessments",
  "job_url": "https://example.com/job/123"
}
```
*(Provide at least one of `query` or `job_url`)*

**Response:**
```json
{
  "recommended_assessments": [
    {
      "url": "https://www.shl.com/solutions/products/product-catalog/view/core-java-entry-level-new/",
      "adaptive_support": "No",
      "description": "Assess Java programming skills...",
      "duration": 35,
      "remote_support": "Yes",
      "test_type": ["K"]
    }
  ]
}
```

---

## ⚙️ Setup & Running Locally

### Prerequisites
- Python 3.10+
- Node.js 18+
- Gemini API key

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Run data augmentation (ONE TIME — before first deploy)
```bash
set GEMINI_API_KEY=your-key
python scripts/augment_and_embed.py
```
This merges `Gen_AI Dataset.xlsx` with `shl_assessments.csv` and computes any missing embeddings → outputs `shl_assessments_v2.csv`.

### 3. Run local dev server (Terminal 1)
```bash
set GEMINI_API_KEY=your-key
python dev_server.py
```

### 4. Run Next.js frontend (Terminal 2)
```bash
cd frontend
npm install
npm run dev
# Open http://localhost:3000
```

---

## 📊 Evaluation

### Mean Recall@10 vs. Train Set
```bash
set GEMINI_API_KEY=your-key
python evaluation/evaluate.py
```

### Generate Predictions for Test Set
```bash
set GEMINI_API_KEY=your-key
python evaluation/generate_predictions.py
# Output: evaluation/predictions.csv
```

---

## 🚀 Deploying to Vercel

1. Go to [vercel.com](https://vercel.com) → **Add New Project**
2. Import GitHub repo: `Itzadityapandey/Rc_Shl`
3. Set **Root Directory** to `./` (repo root)
4. Add env variable: `GEMINI_API_KEY = <your-key>`
5. Deploy

---

## 📁 Project Structure

```
├── api/
│   ├── health.py              # GET /api/health
│   └── recommend.py           # POST /api/recommend (optimized serverless)
├── frontend/                  # Next.js 15 React app
│   └── src/app/
│       ├── page.tsx           # Home: input + results table
│       ├── about/page.tsx     # About / approach page
│       └── components/Navbar.tsx
├── evaluation/
│   ├── evaluate.py            # Mean Recall@K evaluation
│   ├── generate_predictions.py  # Generates predictions.csv
│   ├── train_set.csv          # 65 labeled Q→URL pairs
│   └── test_set.csv           # 9 unlabeled test queries
├── scripts/
│   └── augment_and_embed.py   # Data merge + vectorization script
├── crawler.py                 # SHL catalog web scraper
├── dev_server.py              # Local Flask dev server
├── shl_assessments.csv        # Scraped catalog (321 assessments)
├── vercel.json                # Vercel deployment config
└── requirements.txt
```

---

## 🛠️ Tech Stack

- **Frontend**: Next.js 15, React, TypeScript
- **Backend**: Python Serverless Functions (Vercel)
- **AI**: Google Gemini — `text-embedding-004` + `gemini-1.5-flash`
- **Search**: Cosine similarity (numpy), balanced personality/cognitive ranking
- **Data**: 321+ SHL individual test solutions scraped from official catalog
