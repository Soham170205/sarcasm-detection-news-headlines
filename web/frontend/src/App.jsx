import { useEffect, useRef, useState } from "react";

const EXAMPLES = [
  "Local man shocked to learn exams require studying",
  "Here is what you need to know about the new tax rules",
  "Nation's dads unveil new thermostat protection plan",
  "University opens registration for the new academic year",
  "Area dad spends entire vacation explaining how to load dishwasher",
  "The best budget phones you can buy this year",
];

async function api(path, options) {
  const res = await fetch(`/api${path}`, options);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "Request failed");
  return data;
}

function Meter({ notSarcastic }) {
  return (
    <div className="meter">
      <div className="meter-track">
        <div className="meter-fill" style={{ width: `${notSarcastic}%` }} />
        <div className="meter-mid" />
      </div>
      <div className="meter-labels">
        <span>Not sarcastic</span>
        <span>Sarcastic</span>
      </div>
    </div>
  );
}

function HighlightedHeadline({ words }) {
  const max = Math.max(...words.map((w) => Math.abs(w.score)), 0.001);
  return (
    <p className="highlight">
      {words.map((w, i) => {
        const strength = Math.min(Math.abs(w.score) / max, 1);
        const color =
          w.score > 0
            ? `rgba(255, 107, 74, ${0.15 + strength * 0.65})`
            : `rgba(56, 178, 172, ${0.15 + strength * 0.65})`;
        return (
          <span
            key={i}
            className={`word ${w.known ? "" : "unknown"}`}
            style={w.known ? { background: color } : undefined}
            title={w.known ? `weight ${w.score.toFixed(2)}` : "not in vocabulary"}
          >
            {w.word}
          </span>
        );
      })}
    </p>
  );
}

function Result({ result }) {
  const sarc = result.is_sarcastic;
  return (
    <section className={`card result ${sarc ? "is-sarc" : "is-real"}`}>
      <div className="result-head">
        <div>
          <div className="eyebrow">Verdict</div>
          <h2 className="verdict">{sarc ? "Sarcastic 🙃" : "Not sarcastic 📰"}</h2>
        </div>
        <div className="confidence">
          <div className="big">{result.confidence.toFixed(1)}%</div>
          <div className="eyebrow">confidence</div>
        </div>
      </div>

      <Meter value={result.sarcastic_probability} />
      <div className="probs">
        <span>Not sarcastic: <b>{result.not_sarcastic_probability}%</b></span>
        <span>Sarcastic: <b>{result.sarcastic_probability}%</b></span>
      </div>

      <div className="eyebrow mt">Why? Word influence</div>
      <HighlightedHeadline words={result.word_scores} />
      <div className="legend">
        <span><i className="dot sarc" /> pushes toward sarcastic</span>
        <span><i className="dot real" /> pushes toward genuine</span>
        <span><i className="dot unk" /> unknown to model</span>
      </div>
      {result.known_words < result.total_words / 2 && (
        <p className="warn">
          Only {result.known_words}/{result.total_words} words are in the model's vocabulary —
          this prediction is mostly a guess.
        </p>
      )}

      <div className="eyebrow mt">Top contributing terms (TF-IDF × coefficient)</div>
      <ul className="terms">
        {result.top_terms.map((t) => (
          <li key={t.term}>
            <code>{t.term}</code>
            <span className={t.contribution > 0 ? "pos" : "neg"}>
              {t.contribution > 0 ? "+" : ""}
              {t.contribution.toFixed(3)}
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}

export default function App() {
  const [headline, setHeadline] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [features, setFeatures] = useState(null);
  const [info, setInfo] = useState(null);
  const [online, setOnline] = useState(null);
  const inputRef = useRef(null);

  const loadHistory = () => api("/history").then(setHistory).catch(() => {});

  useEffect(() => {
    api("/health").then(() => setOnline(true)).catch(() => setOnline(false));
    api("/model-info").then(setInfo).catch(() => {});
    api("/top-features?n=12").then(setFeatures).catch(() => {});
    loadHistory();
  }, []);

  const analyze = async (text = headline) => {
    if (!text.trim()) return;
    setHeadline(text);
    setLoading(true);
    setError("");
    try {
      const data = await api("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ headline: text }),
      });
      setResult(data);
      loadHistory();
    } catch (e) {
      setError(e.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const clearHistory = async () => {
    await api("/history", { method: "DELETE" });
    setHistory([]);
  };

  return (
    <div className="page">
      <header className="top">
        <div>
          <h1>Sarcasm<span>Detector</span></h1>
          <p className="sub">TF-IDF + Logistic Regression on 26,709 news headlines</p>
        </div>
        <div className={`status ${online ? "on" : online === false ? "off" : ""}`}>
          <i /> {online ? "Model online" : online === false ? "Backend offline" : "Connecting…"}
        </div>
      </header>

      {info && (
        <div className="stats">
          <div><b>{info.accuracy}%</b><span>test accuracy</span></div>
          <div><b>{info.total_headlines.toLocaleString()}</b><span>headlines</span></div>
          <div><b>{info.precision_sarcastic}%</b><span>precision (sarcastic)</span></div>
          <div><b>10k</b><span>TF-IDF features</span></div>
        </div>
      )}

      <main className="grid">
        <div className="col">
          <section className="card">
            <label className="eyebrow" htmlFor="h">Enter a news headline</label>
            <textarea
              id="h"
              ref={inputRef}
              rows={3}
              maxLength={300}
              value={headline}
              placeholder="e.g. Area man heroically finishes assignment 4 minutes before deadline"
              onChange={(e) => setHeadline(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  analyze();
                }
              }}
            />
            <div className="row">
              <span className="count">{headline.length}/300 · Enter to analyze</span>
              <button onClick={() => analyze()} disabled={loading || !headline.trim()}>
                {loading ? "Analyzing…" : "Analyze"}
              </button>
            </div>

            <div className="eyebrow mt">Try an example</div>
            <div className="chips">
              {EXAMPLES.map((ex) => (
                <button key={ex} className="chip" onClick={() => analyze(ex)}>
                  {ex}
                </button>
              ))}
            </div>
            {error && <p className="error">{error}</p>}
          </section>

          {result && <Result result={result} />}
        </div>

        <div className="col">
          {features && (
            <section className="card">
              <div className="eyebrow">What the model learned</div>
              <div className="feat-cols">
                <div>
                  <h3 className="sarc-t">Sarcasm signals</h3>
                  {features.sarcastic.map((f) => (
                    <div className="feat" key={f.term}>
                      <code>{f.term}</code>
                      <div className="bar sarc" style={{ width: `${(f.weight / features.sarcastic[0].weight) * 100}%` }} />
                    </div>
                  ))}
                </div>
                <div>
                  <h3 className="real-t">Genuine-news signals</h3>
                  {features.not_sarcastic.map((f) => (
                    <div className="feat" key={f.term}>
                      <code>{f.term}</code>
                      <div className="bar real" style={{ width: `${(f.weight / features.not_sarcastic[0].weight) * 100}%` }} />
                    </div>
                  ))}
                </div>
              </div>
            </section>
          )}

          <section className="card">
            <div className="row">
              <div className="eyebrow">Recent predictions</div>
              {history.length > 0 && (
                <button className="ghost" onClick={clearHistory}>Clear</button>
              )}
            </div>
            {history.length === 0 ? (
              <p className="muted">No predictions yet.</p>
            ) : (
              <ul className="history">
                {history.map((h, i) => (
                  <li key={i} onClick={() => analyze(h.Headline)}>
                    <span className={`tag ${h.Prediction === "SARCASTIC" ? "sarc" : "real"}`}>
                      {h.Prediction === "SARCASTIC" ? "SARC" : "REAL"}
                    </span>
                    <span className="h-text">{h.Headline}</span>
                    <span className="h-conf">{h.Confidence}</span>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </div>
      </main>

      <footer>Built with React + Flask · scikit-learn model</footer>
    </div>
  );
}
