import type { Metadata } from "next";

export const metadata: Metadata = {
    title: "About · SHL Assessment Recommender",
    description: "Learn about the approach, data pipeline, and technology stack behind the SHL Assessment Recommender.",
};

export default function AboutPage() {
    return (
        <div className="about-page">
            <span style={{ fontSize: "0.85rem", color: "var(--text-muted)", letterSpacing: "0.06em", textTransform: "uppercase", fontWeight: 600 }}>
                Project Overview
            </span>
            <h1>About This System</h1>
            <p>
                The SHL Assessment Recommender is an AI-powered tool that helps hiring managers and
                recruiters find the most relevant SHL assessments for any role — using semantic
                understanding of job descriptions, not just keyword matching.
            </p>

            <h2>The Problem</h2>
            <p>
                Hiring teams waste significant time manually browsing the SHL catalog of 377+
                individual test solutions to find assessments that match a role&apos;s requirements.
                Keyword filters miss context — a search for &quot;Java developer&quot; won&apos;t surface
                communication or personality assessments that are equally important for success.
            </p>

            <h2>Our Approach</h2>
            <div className="about-grid">
                <div className="about-card">
                    <div className="about-card-icon">🕷️</div>
                    <h3>Data Pipeline</h3>
                    <p>Scraped 377+ assessments from SHL&apos;s product catalog. Each assessment was embedded using Gemini text-embedding-004 and stored as a vector.</p>
                </div>
                <div className="about-card">
                    <div className="about-card-icon">🧠</div>
                    <h3>LLM Query Expansion</h3>
                    <p>Gemini Flash enriches the raw query by extracting key skills and competency needs, dramatically boosting semantic recall.</p>
                </div>
                <div className="about-card">
                    <div className="about-card-icon">⚡</div>
                    <h3>Fast Vector Search</h3>
                    <p>All embeddings are cached in-memory on the server. Cosine similarity over numpy arrays completes in under 50ms.</p>
                </div>
                <div className="about-card">
                    <div className="about-card-icon">⚖️</div>
                    <h3>Balanced Ranking</h3>
                    <p>Results intelligently balance Personality (P) and Cognitive/Knowledge (A, K, B) assessments to cover both hard and soft skills.</p>
                </div>
            </div>

            <h2>Technology Stack</h2>
            <ul>
                <li><strong>Frontend:</strong> Next.js 15, React, TypeScript, CSS custom properties</li>
                <li><strong>Backend:</strong> Python Serverless Functions (Vercel), Flask-compatible handlers</li>
                <li><strong>AI:</strong> Google Gemini — text-embedding-004 (embeddings), Gemini 1.5 Flash (query expansion)</li>
                <li><strong>Search:</strong> Cosine similarity over numpy float32 arrays (no external vector DB needed)</li>
                <li><strong>Data:</strong> 377+ SHL individual test solutions scraped from the official product catalog</li>
                <li><strong>Evaluation:</strong> Mean Recall@10 computed against 10 labeled training queries</li>
            </ul>

            <h2>Evaluation</h2>
            <p>
                The system is evaluated using <strong>Mean Recall@10</strong> — the fraction of relevant
                assessments retrieved in the top 10 results, averaged across all test queries. The balanced
                ranking strategy ensures that multi-domain queries (e.g., a technical + interpersonal role)
                receive a representative mix of assessment types.
            </p>

            <h2>API Endpoints</h2>
            <ul>
                <li><code>GET /api/health</code> — Returns <code>{"{"}"status": "healthy"{"}"}</code></li>
                <li><code>POST /api/recommend</code> — Accepts <code>{"{"}"query": "..."{"}"}</code> or <code>{"{"}"job_url": "..."{"}"}</code>, returns up to 10 ranked assessments in JSON format</li>
            </ul>

            <footer className="footer" style={{ marginTop: "3rem", padding: 0, border: "none" }}>
                Built for the SHL Gen AI Hiring Challenge · 2025
            </footer>
        </div>
    );
}
