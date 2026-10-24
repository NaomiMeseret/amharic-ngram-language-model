"""Extrinsic evaluation: next-word prediction and real-vs-shuffled discrimination."""
from __future__ import annotations
import random
import numpy as np
from src.ngram_model import NGramModel, UNK, EOS, PAD, BOS

SPECIAL = {PAD, UNK, BOS, EOS}

def _top_candidates(model: NGramModel, context: list[int], order: int, k: int):
    """Exact-order candidates ranked by count (fast: no full-vocab scan)."""
    if order == 1:
        oc = model.orders[1]
        idx = np.argsort(oc.counts)[::-1][: k + len(SPECIAL)]
        ids = [int(oc.keys[i]) for i in idx if int(oc.keys[i]) not in SPECIAL]
        return ids[:k]
    ctx = context[-(order - 1):]
    pairs = [(t, c) for t, c in model.orders[order].next_counts(ctx) if t not in SPECIAL]
    pairs.sort(key=lambda p: -p[1])
    return [t for t, _ in pairs[:k]]

def next_word_eval(model, sentences, order, ks=(1, 3, 5)):
    """sentences: iterable of [BOS, ..., EOS] ID lists."""
    hits = dict.fromkeys(ks, 0)
    n = covered = skipped = 0
    for sent in sentences:
        padded = [BOS] * (order - 1) + sent[1:]          # same framing as training
        for i in range(order - 1, len(padded)):
            target = padded[i]
            if target in SPECIAL:                         # UNK/EOS are not suggestable
                skipped += 1
                continue
            n += 1
            cands = _top_candidates(model, padded[:i], order, max(ks))
            if not cands:
                continue
            covered += 1
            for k in ks:
                hits[k] += target in cands[:k]
    out = {f"top{k}": hits[k] / n for k in ks}
    out |= {"coverage": covered / n, "evaluated": n, "skipped_unk_eos": skipped}
    return out

def sentence_logprob(model, sent, order, method="backoff"):
    padded = [BOS] * (order - 1) + sent[1:]
    return sum(model.logprob(padded[i], padded[i - order + 1:i], method=method)
               for i in range(order - 1, len(padded)))

def discrimination_eval(model, sentences, order, seed=0, method="backoff"):
    rnd, wins, ties, n = random.Random(seed), 0, 0, 0
    for sent in sentences:
        body = sent[1:-1]
        if len(set(body)) < 2:
            continue
        shuffled = body[:]
        for _ in range(10):                               # avoid identical shuffles
            rnd.shuffle(shuffled)
            if shuffled != body:
                break
        else:
            continue
        a = sentence_logprob(model, sent, order, method)
        b = sentence_logprob(model, [BOS, *shuffled, EOS], order, method)
        n += 1
        wins += a > b
        ties += a == b
    return {"accuracy": wins / n, "ties": ties, "n": n}