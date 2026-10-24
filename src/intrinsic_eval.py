import math
import numpy as np

from src.ngram_model import BOS, EOS, UNK

K_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]
INTERP_WEIGHTS = [0.2, 0.4, 0.1, 0.3]  # tuned by the team on dev (full model)


# ----------------------------------------------------------------- data helpers
def sample_sentences(ids, n_sent, seed=42):
    """Random subset of complete <s> ... </s> sentences (all of them if n_sent is None)."""
    ids = np.asarray(ids)
    starts = np.flatnonzero(ids == BOS)
    ends = np.flatnonzero(ids == EOS)
    assert len(starts) == len(ends), "malformed sentence stream"
    if n_sent is None or n_sent >= len(starts):
        return ids
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(starts), n_sent, replace=False))
    return np.concatenate([ids[starts[i]:ends[i] + 1] for i in pick])


def remove_stopwords(ids, vocab_path, stopwords_path):
    """Same filtering that was used to build the no-stopword training data."""
    ids = np.asarray(ids)
    vocab = open(vocab_path, encoding="utf-8").read().splitlines()
    w2i = {w: i for i, w in enumerate(vocab)}
    stops = [w2i[w.strip()] for w in open(stopwords_path, encoding="utf-8").read().splitlines()
             if w.strip() in w2i and w2i[w.strip()] >= 4]
    is_stop = np.zeros(len(vocab), dtype=bool)
    is_stop[stops] = True
    f = ids[~is_stop[ids]]
    bos = np.flatnonzero(f == BOS)
    empty = bos[f[bos + 1] == EOS]            # sentences that became empty
    drop = np.zeros(len(f), dtype=bool)
    drop[empty] = True
    drop[empty + 1] = True
    return f[~drop]


# ------------------------------------------------------------ vectorised counts
def build_events(ids, order):
    """For every word/</s> in the stream: its n-1 context ids (BOS-padded) and its id."""
    ids = np.asarray(ids, dtype=np.int64)
    pos = np.arange(len(ids))
    is_bos = ids == BOS
    sent_start = np.maximum.accumulate(np.where(is_bos, pos, 0))
    target_idx = np.flatnonzero(~is_bos)
    ctx = []
    for d in range(order - 1, 0, -1):          # oldest context word first
        ctx.append(ids[np.maximum(target_idx - d, sent_start[target_idx])])
    return ctx, ids[target_idx]


def _pack(cols):
    key = np.zeros(len(cols[0]), dtype=np.uint64)
    for c in cols:
        key = (key << np.uint64(16)) | c.astype(np.uint64)
    return key


def _lookup(sorted_keys, values, query, chunk=1_000_000):
    out = np.zeros(len(query), dtype=np.float64)
    n = len(sorted_keys)
    for s in range(0, len(query), chunk):
        q = query[s:s + chunk]
        idx = np.searchsorted(sorted_keys, q)
        idx_c = np.minimum(idx, n - 1)
        hit = (idx < n) & (sorted_keys[idx_c] == q)
        part = np.zeros(len(q), dtype=np.float64)
        part[hit] = values[idx_c[hit]]
        out[s:s + chunk] = part
    return out


def event_counts(model, ids, order):
    """Return (count of n-gram, count of its context) for every prediction event."""
    o = model.orders[order]
    ctx, targets = build_events(ids, order)
    c = _lookup(o.keys, o.counts, _pack(ctx + [targets]))
    if order == 1:
        d = np.full(len(c), float(model.total_occurrences(1)))
    else:
        d = _lookup(o.context_keys, o.context_totals, _pack(ctx))
    return c, d


def ppl_addk(c, d, k, V):
    p = (c + k) / (d + k * V)
    return float(math.exp(-np.mean(np.log(p))))


def ppl_mle(c, d):
    zero = c == 0
    frac = float(zero.mean())
    ppl_seen = float(math.exp(-np.mean(np.log(c[~zero] / d[~zero]))))
    return (float("inf") if frac > 0 else ppl_seen), frac, ppl_seen


def tune_add_k(model, dev_ids, grid=K_GRID):
    """Best add-k per order on the DEVELOPMENT data (never the test set)."""
    V = len(model.allowed_outcomes)
    best = {}
    for n in sorted(model.orders):
        c, d = event_counts(model, dev_ids, n)
        scores = {k: ppl_addk(c, d, k, V) for k in grid}
        best[n] = min(scores, key=scores.get)
    return best


# --------------------------------------------------------- slow (repo prob()) ppl
def slow_ppl(model, ids, order, method, weights=None):
    """Perplexity through model.prob (backoff / interpolation). Slow: use a small sample."""
    ctx, targets = build_events(ids, order)
    total = 0.0
    for j in range(len(targets)):
        context = [int(col[j]) for col in ctx]
        p = model.prob(int(targets[j]), context, method=method, weights=weights)
        if p <= 0:
            return float("inf")
        total += math.log(p)
    return float(math.exp(-total / len(targets)))


# ------------------------------------------------------------------- evaluation
def evaluate_model(model, test_ids, dev_ids, label, n_test=50_000, n_dev=5_000,
                   n_slow=1_000, seed=42, do_slow=True):
    """Perplexity table for n=1..4. test_ids/dev_ids are flat <s>..</s> id streams."""
    import pandas as pd
    V = len(model.allowed_outcomes)
    dev = sample_sentences(dev_ids, n_dev, seed)
    test = sample_sentences(test_ids, n_test, seed)
    slow = sample_sentences(test_ids, n_slow, seed + 1)
    best_k = tune_add_k(model, dev)
    rows = []
    for n in sorted(model.orders):
        c, d = event_counts(model, test, n)
        mle, zero_frac, mle_seen = ppl_mle(c, d)
        row = {"model": label, "n": n, "events": len(c),
               "MLE": mle, "MLE_zero_prob_%": 100 * zero_frac,
               "best_k(dev)": best_k[n], "add_k": ppl_addk(c, d, best_k[n], V),
               "add_1": ppl_addk(c, d, 1.0, V)}
        if do_slow:
            row["backoff(sample)"] = slow_ppl(model, slow, n, "backoff")
            if n == 4:
                row["interp(sample)"] = slow_ppl(model, slow, 4, "interpolation", INTERP_WEIGHTS)
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- generation
def generate(model, order, n_sent=5, max_len=30, seed=0, allow_unk=False):
    """Random sentences from the order-n model, sampling from the observed counts.

    If the current n-1 word context was never seen, shorter contexts are used for that step.
    Returns (sentences, share_of_steps_that_needed_a_shorter_context).
    """
    rng = np.random.default_rng(seed)
    vocab = model.vocabulary
    o1 = model.orders[1]
    uni_ids = np.asarray(o1.keys, dtype=np.int64)
    uni_cnt = np.asarray(o1.counts, dtype=np.float64)
    if not allow_unk:
        uni_cnt = np.where(uni_ids == UNK, 0.0, uni_cnt)
    steps = shorter = 0

    def next_id(ctx):
        nonlocal shorter
        for m in range(order - 1, 0, -1):
            pairs = model.orders[m + 1].next_counts(ctx[len(ctx) - m:])
            if not allow_unk:
                pairs = [(t, c) for t, c in pairs if t != UNK]
            if pairs:
                if m < order - 1:
                    shorter += 1
                t = np.array([p[0] for p in pairs])
                w = np.array([p[1] for p in pairs], dtype=np.float64)
                return int(rng.choice(t, p=w / w.sum()))
        if order > 1:
            shorter += 1
        return int(rng.choice(uni_ids, p=uni_cnt / uni_cnt.sum()))

    out = []
    for _ in range(n_sent):
        ctx = [BOS] * (order - 1)
        words, ended = [], False
        for _ in range(max_len):
            t = next_id(ctx)
            steps += 1
            if t == EOS:
                ended = True
                break
            words.append(vocab[t])
            ctx = (ctx + [t])[-(order - 1):] if order > 1 else []
        out.append(" ".join(words) + ("" if ended else " ..."))
    return out, (shorter / steps if steps else 0.0)
