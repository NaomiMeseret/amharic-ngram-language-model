import { useEffect, useRef, useState } from 'react';

const API_URL = 'http://127.0.0.1:8000';
const SAMPLE_TEXT = 'ኢትዮጵያ በአፍሪካ';

function BrandMark({ className = '' }) {
  return (
    <span className={`brand-mark ${className}`.trim()} aria-hidden="true" lang="am">አ</span>
  );
}

function ArrowIcon() {
  return (
    <svg viewBox="0 0 20 20" aria-hidden="true">
      <path d="M3.5 10h12m-4.5-4.5L15.5 10 11 14.5" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function SparkIcon() {
  return (
    <svg viewBox="0 0 20 20" aria-hidden="true">
      <path d="M10 2.8 11.7 8l5.2 1.9-5.2 1.9L10 17l-1.7-5.2L3.1 9.9 8.3 8 10 2.8Z" fill="none" stroke="currentColor" strokeWidth="1.25" strokeLinejoin="round" />
    </svg>
  );
}

function InfoIcon() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <circle cx="8" cy="8" r="6.5" fill="none" stroke="currentColor" strokeWidth="1.1" />
      <path d="M8 7.3v3.4M8 5.1v.2" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  );
}

function App() {
  const textareaRef = useRef(null);
  const [text, setText] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [apiStatus, setApiStatus] = useState('checking');
  const [generatedText, setGeneratedText] = useState('');
  const [generating, setGenerating] = useState(false);
  const [generationError, setGenerationError] = useState('');

  useEffect(() => {
    let active = true;

    const checkApi = async () => {
      try {
        const response = await fetch(`${API_URL}/health`);
        if (!response.ok) throw new Error('API health check failed');
        if (active) setApiStatus('online');
      } catch {
        if (active) setApiStatus('offline');
      }
    };

    checkApi();
    const interval = window.setInterval(checkApi, 30000);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    const trimmed = text.trim();
    if (!trimmed) {
      setSuggestions([]);
      setLoading(false);
      setError('');
      return undefined;
    }

    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      setLoading(true);
      setError('');
      try {
        const response = await fetch(`${API_URL}/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: trimmed, top_k: 5 }),
          signal: controller.signal,
        });
        if (!response.ok) {
          const detail = await response.text();
          throw new Error(detail || 'Prediction request failed.');
        }
        const data = await response.json();
        setSuggestions(Array.isArray(data.predictions) ? data.predictions : []);
        setApiStatus('online');
      } catch (requestError) {
        if (requestError.name !== 'AbortError') {
          setSuggestions([]);
          setError('Could not reach the prediction service. Check that the backend is running, then try again.');
          setApiStatus('offline');
        }
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }, 260);

    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [text]);

  const insertSuggestion = (word) => {
    const value = text.trimEnd();
    const nextValue = value ? `${value} ${word}` : word;
    setText(nextValue);
    setGeneratedText('');
    requestAnimationFrame(() => {
      const element = textareaRef.current;
      if (!element) return;
      element.focus();
      element.selectionStart = element.selectionEnd = nextValue.length;
    });
  };

  const handleKeyDown = (event) => {
    if (!suggestions.length) return;
    if (event.key === 'Tab') {
      event.preventDefault();
      insertSuggestion(suggestions[0].word);
      return;
    }
    if (/^[1-5]$/.test(event.key)) {
      const suggestion = suggestions[Number(event.key) - 1];
      if (suggestion) {
        event.preventDefault();
        insertSuggestion(suggestion.word);
      }
    }
  };

  const generateContinuation = async () => {
    setGenerating(true);
    setGenerationError('');
    setGeneratedText('');
    try {
      const response = await fetch(`${API_URL}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: text.trim(), max_tokens: 5 }),
      });
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || 'Generation request failed.');
      }
      const data = await response.json();
      setGeneratedText(data.generated_text || '');
      setApiStatus('online');
    } catch {
      setGenerationError('Could not generate a continuation. Check that the backend is running.');
      setApiStatus('offline');
    } finally {
      setGenerating(false);
    }
  };

  const clearComposer = () => {
    setText('');
    setSuggestions([]);
    setGeneratedText('');
    setError('');
    setGenerationError('');
    textareaRef.current?.focus();
  };

  const statusText = {
    checking: 'Checking service',
    online: 'Model service online',
    offline: 'Model service offline',
  }[apiStatus];

  const topProbability = suggestions.reduce(
    (highest, item) => Math.max(highest, Number(item.prob ?? 0)),
    0,
  );

  return (
    <div className="app-shell" id="top">
      <aside className="sidebar" aria-label="Workspace navigation">
        <a className="brand" href="#top" aria-label="Amharic N-gram Lab home">
          <BrandMark />
          <span className="brand-text">
            <strong>Amharic</strong>
            <small>N-Gram Lab</small>
          </span>
        </a>

        <p className="rail-label">Workspace</p>
        <nav className="rail-nav">
          <a className="rail-link is-active" href="#composer">
            <span className="nav-index" aria-hidden="true">01</span>
            <span className="nav-copy">
              <strong>Word suggestions</strong>
              <small>Ranked next-word candidates</small>
            </span>
          </a>
          <a className="rail-link" href="#model-notes">
            <span className="nav-index" aria-hidden="true">02</span>
            <span className="nav-copy">
              <strong>Model notes</strong>
              <small>Orders, vocabulary, strategy</small>
            </span>
          </a>
        </nav>

        <div className="rail-footer">
          <p className="rail-language">
            <span className="language-dot" aria-hidden="true" />
            <span>Amharic · <span lang="am">አማርኛ</span></span>
          </p>
          <p className="rail-runtime">Local API</p>
        </div>
      </aside>

      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumb">
            <span>Project</span>
            <span className="crumb-slash" aria-hidden="true">/</span>
            <strong>Demo</strong>
          </div>
          <div className={`service-state ${apiStatus}`} role="status" aria-live="polite">
            <span className="service-indicator" />
            <span className="service-label">{statusText}</span>
          </div>
        </header>

        <main className="content">
          <section className="hero" aria-labelledby="page-title">
            <div className="eyebrow">
              <span className="eyebrow-rule" aria-hidden="true" />
              <span>Language model interface</span>
              <span className="eyebrow-tag">Local demo</span>
            </div>
            <h1 id="page-title">
              Write a thought.<span className="title-second">Find its next word.</span>
            </h1>
            <p className="hero-copy">Amharic next-word suggestions, powered by N-gram models.</p>
          </section>

          <section className="work-grid" id="composer" aria-label="Next-word suggestion workspace">
            <article className="panel composer">
              <header className="panel-heading">
                <div>
                  <p className="section-kicker">01 / Compose</p>
                  <h2>Continue your sentence</h2>
                </div>
                <span className="method-chip">
                  <span className="chip-dot" aria-hidden="true" />
                  Backoff
                </span>
              </header>

              <label className="sr-only" htmlFor="sentence-input">Type an Amharic sentence or phrase</label>
              <div className="input-frame">
                <textarea
                  ref={textareaRef}
                  id="sentence-input"
                  value={text}
                  onChange={(event) => { setText(event.target.value); setGeneratedText(''); }}
                  onKeyDown={handleKeyDown}
                  placeholder="እዚህ ይጻፉ…"
                  className="sentence-input"
                  rows={5}
                  spellCheck="false"
                  dir="auto"
                  aria-describedby="composer-hint"
                />
              </div>

              <div className="composer-footer">
                <p id="composer-hint" className="keyboard-hint">
                  <kbd>Tab</kbd> first suggestion
                  <span className="hint-separator" aria-hidden="true">/</span>
                  <kbd>1–5</kbd> choose by rank
                </p>
                <button className="text-button" type="button" onClick={clearComposer} disabled={!text && !generatedText}>
                  Clear text
                </button>
              </div>

              <section className="suggestion-section" aria-labelledby="suggestion-title">
                <div className="suggestion-heading">
                  <div>
                    <p className="section-kicker">Model output</p>
                    <h3 id="suggestion-title">Next-word candidates</h3>
                  </div>
                  {loading && (
                    <span className="loading-label">
                      <span className="spinner" aria-hidden="true" />
                      Updating
                    </span>
                  )}
                  {!loading && suggestions.length > 0 && (
                    <span className="result-count">{suggestions.length} results</span>
                  )}
                </div>

                {error ? (
                  <div className="notice error-notice" role="alert">
                    <strong>Prediction unavailable</strong>
                    <span>{error}</span>
                  </div>
                ) : !text.trim() ? (
                  <div className="empty-state">
                    <span className="empty-glyph" aria-hidden="true" lang="am">አ</span>
                    <div className="empty-copy">
                      <strong>Type a phrase to see suggestions.</strong>
                      <span>Suggestions appear as you write.</span>
                    </div>
                    <button className="sample-link" type="button" onClick={() => setText(SAMPLE_TEXT)}>
                      Try an example <ArrowIcon />
                    </button>
                  </div>
                ) : loading && suggestions.length === 0 ? (
                  <div className="candidate-list" aria-label="Loading suggestions">
                    {[0, 1, 2].map((item) => <div className="candidate-skeleton" key={item} />)}
                  </div>
                ) : suggestions.length === 0 ? (
                  <div className="empty-state compact-empty">
                    <div className="empty-copy">
                      <strong>No candidates returned.</strong>
                      <span>Try a shorter or different phrase.</span>
                    </div>
                  </div>
                ) : (
                  <ol className="candidate-list" aria-label="Ranked word suggestions">
                    {suggestions.map((item, index) => {
                      const probability = Math.max(0, Number(item.prob ?? 0));
                      const share = topProbability > 0 ? (probability / topProbability) * 100 : 0;
                      return (
                        <li key={`${item.word}-${index}`} className="candidate-item">
                          <button
                            className="candidate-row"
                            type="button"
                            onClick={() => insertSuggestion(item.word)}
                            aria-label={`Insert ${item.word}, rank ${index + 1}, probability ${(probability * 100).toFixed(2)} percent`}
                            style={{ animationDelay: `${index * 45}ms` }}
                          >
                            <span className="candidate-rank">{String(index + 1).padStart(2, '0')}</span>
                            <span className="candidate-main">
                              <span className="candidate-word" lang="am">{item.word}</span>
                              <span className="candidate-meter" aria-hidden="true">
                                <span className="candidate-meter-fill" style={{ width: `${Math.max(2, share)}%` }} />
                              </span>
                            </span>
                            <span className="candidate-probability">
                              {(probability * 100).toFixed(2)}<small>%</small>
                            </span>
                            <span className="candidate-action" aria-hidden="true"><ArrowIcon /></span>
                          </button>
                        </li>
                      );
                    })}
                  </ol>
                )}
                <p className="score-note">Scores are model conditional probabilities.</p>
              </section>

              <div className="generation-block">
                <div className="generation-copy">
                  <span className="generation-icon" aria-hidden="true"><SparkIcon /></span>
                  <span className="generation-text">
                    <strong>Generate a continuation</strong>
                    <small>Continue with the top-ranked words.</small>
                  </span>
                </div>
                <button
                  className="generate-button"
                  type="button"
                  onClick={generateContinuation}
                  disabled={generating || apiStatus === 'offline'}
                >
                  {generating ? 'Generating…' : 'Generate'}
                  <ArrowIcon />
                </button>
              </div>
              {generationError && <p className="inline-error" role="alert">{generationError}</p>}
              {generatedText && (
                <div className="generated-result" aria-live="polite">
                  <p className="section-kicker">Generated text</p>
                  <p className="generated-text" lang="am" dir="auto">{generatedText}</p>
                  <button className="text-button" type="button" onClick={() => setText(generatedText)}>
                    Continue editing <ArrowIcon />
                  </button>
                </div>
              )}
            </article>

            <aside className="info-column">
              <section className="panel model-card" id="model-notes">
                <p className="section-kicker">02 / Model profile</p>
                <h2>A compact model.<br />A large language.</h2>
                <p className="model-summary">Sparse unigram through four-gram models for Amharic.</p>
                <div className="stat-grid">
                  <div className="stat-cell">
                    <strong>01–04</strong>
                    <span>Gram orders</span>
                  </div>
                  <div className="stat-cell">
                    <strong>50,000</strong>
                    <span>Vocabulary</span>
                  </div>
                </div>
                <div className="profile-row"><span>Prediction strategy</span><strong>Backoff</strong></div>
                <div className="profile-row"><span>Suggestion limit</span><strong>Top five</strong></div>
                <p className="profile-footnote">Backoff prediction · no model-order switch.</p>
              </section>

              <section className="panel guide-card" aria-labelledby="guide-title">
                <div className="guide-topline">
                  <p className="section-kicker">Quick guide</p>
                  <span className="guide-mark" aria-hidden="true">03</span>
                </div>
                <h2 id="guide-title">How to use</h2>
                <ol className="guide-steps">
                  <li>
                    <span className="step-number" aria-hidden="true">1</span>
                    <p><strong>Write a phrase</strong><small>Type in Amharic; suggestions follow the cursor.</small></p>
                  </li>
                  <li>
                    <span className="step-number" aria-hidden="true">2</span>
                    <p><strong>Choose a candidate</strong><small>Click a row, press Tab, or use keys 1–5.</small></p>
                  </li>
                  <li>
                    <span className="step-number" aria-hidden="true">3</span>
                    <p><strong>Continue writing</strong><small>The model re-ranks with every new word.</small></p>
                  </li>
                </ol>
              </section>

              <div className="backend-note">
                <span className="backend-note-icon" aria-hidden="true"><InfoIcon /></span>
                <p>Local model service required.</p>
              </div>
            </aside>
          </section>

          <footer className="page-footer">
            <span>Amharic N-Gram Language Model</span>
            <span>Research demonstration <i aria-hidden="true" /> Local API</span>
          </footer>
        </main>
      </div>
    </div>
  );
}

export default App;
