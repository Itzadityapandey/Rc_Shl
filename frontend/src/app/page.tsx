"use client";

import { useState } from "react";

/* ─── Types ────────────────────────────────────────────────── */
interface Assessment {
  url: string;
  adaptive_support: string;
  description: string;
  duration: number;
  remote_support: string;
  test_type: string[];
}

/* ─── Badge helper ──────────────────────────────────────────── */
const TYPE_LABELS: Record<string, string> = {
  P: "Personality",
  A: "Ability",
  K: "Knowledge",
  B: "Biodata",
  S: "Simulation",
  C: "Competency",
};

function TypeBadge({ code }: { code: string }) {
  const cls = `badge badge-${["P", "A", "K", "B", "S", "C"].includes(code) ? code : "default"}`;
  return <span className={cls}>{TYPE_LABELS[code] ?? code}</span>;
}

function SupportPill({ value }: { value: string }) {
  const yes = value?.toLowerCase() === "yes";
  return (
    <span className={`pill ${yes ? "pill-yes" : "pill-no"}`}>
      {yes ? "✓ Yes" : "✗ No"}
    </span>
  );
}

/* ─── Skeleton row ──────────────────────────────────────────── */
function SkeletonRows() {
  return (
    <>
      {Array.from({ length: 8 }).map((_, i) => (
        <tr key={i}>
          <td colSpan={6}>
            <div className="skeleton skeleton-row" />
          </td>
        </tr>
      ))}
    </>
  );
}

/* ─── Main Page ─────────────────────────────────────────────── */
export default function Home() {
  const [mode, setMode] = useState<"text" | "url">("text");
  const [query, setQuery] = useState("");
  const [jobUrl, setJobUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<Assessment[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    if (mode === "text" && !query.trim()) return;
    if (mode === "url" && !jobUrl.trim()) return;

    setLoading(true);
    setError(null);
    setResults(null);

    const body =
      mode === "text"
        ? { query: query.trim() }
        : { job_url: jobUrl.trim() };

    try {
      const res = await fetch("/api/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? "Unknown error");
      setResults(data.recommended_assessments ?? []);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  const extractName = (url: string) => {
    try {
      const slug = url.split("/").filter(Boolean).pop() ?? url;
      return slug.replace(/-/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
    } catch {
      return url;
    }
  };

  return (
    <>
      {/* ── Hero ── */}
      <section className="hero">
        <div className="hero-badge">⚡ Powered by Gemini AI</div>
        <h1>Find the Perfect Assessment</h1>
        <p>
          Paste a job description or URL and get AI-powered SHL assessment
          recommendations tailored to your role — in seconds.
        </p>

        {/* ── Input Card ── */}
        <div className="input-card">
          <div className="input-toggle">
            <button
              id="toggle-text"
              className={`toggle-btn ${mode === "text" ? "active" : ""}`}
              onClick={() => { setMode("text"); setResults(null); setError(null); }}
            >
              📝 Job Description
            </button>
            <button
              id="toggle-url"
              className={`toggle-btn ${mode === "url" ? "active" : ""}`}
              onClick={() => { setMode("url"); setResults(null); setError(null); }}
            >
              🔗 Job URL
            </button>
          </div>

          {mode === "text" ? (
            <>
              <label className="input-label" htmlFor="jd-input">
                Paste your job description
              </label>
              <textarea
                id="jd-input"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="e.g., We are looking for a Java developer who can collaborate with business teams…"
              />
            </>
          ) : (
            <>
              <label className="input-label" htmlFor="url-input">
                Job posting URL
              </label>
              <input
                id="url-input"
                type="url"
                value={jobUrl}
                onChange={(e) => setJobUrl(e.target.value)}
                placeholder="https://example.com/jobs/software-engineer"
              />
            </>
          )}

          <button
            id="submit-btn"
            className="submit-btn"
            onClick={handleSubmit}
            disabled={loading}
          >
            {loading ? (
              <>
                <span className="spinner" /> Analyzing &amp; Recommending…
              </>
            ) : (
              "Get Recommendations →"
            )}
          </button>
        </div>
      </section>

      {/* ── Results ── */}
      {(loading || results !== null || error) && (
        <section className="results-section">
          {error && (
            <div className="status-box">
              <div className="icon">⚠️</div>
              <h3>Something went wrong</h3>
              <p>{error}</p>
            </div>
          )}

          {!error && (loading || results !== null) && (
            <>
              {!loading && results !== null && (
                <div className="results-header">
                  <h2 className="results-title">Recommended Assessments</h2>
                  <span className="results-count">{results.length} found</span>
                </div>
              )}

              {!loading && results?.length === 0 && (
                <div className="status-box">
                  <div className="icon">🔍</div>
                  <h3>No results found</h3>
                  <p>Try refining your job description with more detail.</p>
                </div>
              )}

              {(loading || (results && results.length > 0)) && (
                <div className="results-table-wrapper">
                  <table>
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>Assessment Name</th>
                        <th>Test Type</th>
                        <th>Duration</th>
                        <th>Remote</th>
                        <th>Adaptive</th>
                      </tr>
                    </thead>
                    <tbody>
                      {loading ? (
                        <SkeletonRows />
                      ) : (
                        results!.map((a, i) => (
                          <tr key={a.url}>
                            <td style={{ color: "var(--text-muted)", fontWeight: 600 }}>
                              {i + 1}
                            </td>
                            <td>
                              <a
                                href={a.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="assessment-name-link"
                                title={a.description}
                              >
                                {extractName(a.url)}
                              </a>
                            </td>
                            <td>
                              {a.test_type.length > 0
                                ? a.test_type.map((t) => <TypeBadge key={t} code={t} />)
                                : <span style={{ color: "var(--text-muted)" }}>—</span>}
                            </td>
                            <td>
                              {a.duration > 0
                                ? <span style={{ fontWeight: 500 }}>{a.duration} min</span>
                                : <span style={{ color: "var(--text-muted)" }}>N/A</span>}
                            </td>
                            <td><SupportPill value={a.remote_support} /></td>
                            <td><SupportPill value={a.adaptive_support} /></td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              )}
            </>
          )}
        </section>
      )}

      <footer className="footer">
        Built with Next.js + Gemini AI · Data from SHL Assessment Catalog
      </footer>
    </>
  );
}
