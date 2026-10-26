import { useEffect, useRef, useState } from 'react';

const MODEL_ORDER = 'Trigram';
const API_URL = 'http://127.0.0.1:8000';

export default function App() {
  const textareaRef = useRef(null);
  const [text, setText] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    const trimmed = text.trim();

    if (!trimmed) {
      setSuggestions([]);
      setLoading(false);
      return;
    }

    const timer = setTimeout(async () => {
      try {
        setLoading(true);
        setError('');

        const response = await fetch(`${API_URL}/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: trimmed, top_k: 5 }),
        });

        if (!response.ok) {
          const detail = await response.text();
          throw new Error(detail || 'Prediction request failed');
        }

        const data = await response.json();
        setSuggestions(data.predictions ?? []);
      } catch (err) {
        console.error('Prediction failed:', err);
        setError(err.message || 'Could not load suggestions.');
        setSuggestions([]);
      } finally {
        setLoading(false);
      }
    }, 280);

    return () => clearTimeout(timer);
  }, [text]);

  const insertSuggestion = (word) => {
    const value = (text || '').trimEnd();
    const nextValue = value ? `${value} ${word}` : word;
    setText(nextValue);

    requestAnimationFrame(() => {
      const el = textareaRef.current;
      if (!el) return;
      el.focus();
      el.selectionStart = el.selectionEnd = nextValue.length;
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
      event.preventDefault();
      const index = Number(event.key) - 1;
      const choice = suggestions[index];
      if (choice) insertSuggestion(choice.word);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-amber-50 via-white to-orange-50 px-4 py-10 text-stone-800">
      <div className="mx-auto max-w-4xl">
        <div className="rounded-[28px] border border-orange-100 bg-white/90 p-5 shadow-[0_20px_60px_rgba(120,72,40,0.08)] backdrop-blur-sm md:p-8">
          <header className="mb-6 flex items-center justify-between gap-3">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-orange-500">Model</p>
              <h1 className="mt-1 text-2xl font-bold text-stone-800">Amharic Next-Word Predictor</h1>
            </div>

            <div className="rounded-full border border-orange-200 bg-orange-50 px-3 py-1 text-sm font-medium text-orange-700">
              {MODEL_ORDER}
            </div>
          </header>

          <div className="space-y-4">
            <label className="block">
              <span className="mb-2 block text-sm font-medium text-stone-600">Sentence</span>

              <textarea
                ref={textareaRef}
                value={text}
                onChange={(event) => setText(event.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="እኔ ወደ ..."
                className="w-full resize-none rounded-2xl border border-orange-200 bg-white px-4 py-4 text-lg leading-8 text-stone-800 shadow-inner outline-none transition focus:border-orange-400 focus:ring-4 focus:ring-orange-100"
                style={{ fontFamily: '"Noto Sans Ethiopic", "Noto Sans", sans-serif' }}
                rows={6}
              />
            </label>

            <div className="flex items-center justify-between gap-3">
              <div className="text-sm text-stone-500">
                {loading ? (
                  <span className="inline-flex items-center gap-2">
                    <span className="h-2 w-2 animate-pulse rounded-full bg-orange-500" />
                    Predicting...
                  </span>
                ) : !text.trim() ? (
                  'Start typing an Amharic sentence'
                ) : suggestions.length ? (
                  'Suggestions ready'
                ) : (
                  'No continuation found'
                )}
              </div>

              <button
                type="button"
                onClick={() => setText('')}
                className="rounded-xl border border-stone-200 bg-stone-50 px-3 py-2 text-sm font-medium text-stone-700 transition hover:border-stone-300 hover:bg-stone-100"
              >
                Clear
              </button>
            </div>

            {error && (
              <div className="rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                {error}
              </div>
            )}

            <div className="rounded-2xl border border-orange-100 bg-orange-50/50 p-4">
              <div className="mb-3 flex items-center justify-between">
                <p className="text-sm font-semibold text-stone-700">Suggestions</p>
                <p className="text-xs text-stone-500">Tab or 1–5</p>
              </div>

              {!text.trim() ? (
                <p className="text-sm text-stone-500">Enter a phrase to generate next-word suggestions.</p>
              ) : suggestions.length === 0 ? (
                <p className="text-sm text-stone-500">No continuation was found for this text.</p>
              ) : (
                <div className="flex flex-wrap gap-3">
                  {suggestions.map((item, index) => {
                    const percent = Math.max(0, Number(item.prob ?? 0) * 100);
                    return (
                      <button
                        key={`${item.word}-${index}`}
                        type="button"
                        onClick={() => insertSuggestion(item.word)}
                        className="group inline-flex items-center gap-2 rounded-full border border-orange-200 bg-white px-3 py-2 text-base font-medium text-stone-800 shadow-sm transition hover:border-orange-300 hover:bg-orange-50"
                        title={`Insert "${item.word}"`}
                      >
                        <span className="font-semibold" style={{ fontFamily: '"Noto Sans Ethiopic", "Noto Sans", sans-serif' }}>
                          {item.word}
                        </span>
                        <span className="rounded-full bg-orange-100 px-2 py-0.5 text-xs text-orange-700">
                          {percent.toFixed(1)}%
                        </span>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
