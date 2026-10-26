"""Extrinsic evaluation: next-word prediction and real-vs-shuffled discrimination."""
from __future__ import annotations
import math
import random
import numpy as np
from src.ngram_model import NGramModel, UNK, EOS, PAD, BOS

SPECIAL = {PAD, UNK, BOS, EOS}


def make_exact_predictor(model: NGramModel, order: int):
    """Exact-order, no-backoff predictor.

    Returns predict(context_ids, k) -> (candidate_ids, context_seen), where
    context_seen is True if the order-n context occurred in training at all
    (even if it was only ever followed by UNK/EOS).
    """
    if order == 1:
        oc = model.orders[1]
        ranked = [int(oc.keys[i]) for i in np.argsort(oc.counts)[::-1]
                  if int(oc.keys[i]) not in SPECIAL]          # computed once

        def predict(context, k):
            return ranked[:k], True
        return predict

    def predict(context, k):
        raw = list(model.orders[order].next_counts(context[-(order - 1):]))
        pairs = [(t, c) for t, c in raw if t not in SPECIAL]
        pairs.sort(key=lambda p: -p[1])
        return [t for t, _ in pairs[:k]], bool(raw)
    return predict


def next_word_eval(model, sentences, order=None, ks=(1, 3, 5), predictor=None):
    """sentences: iterable of [BOS, ..., EOS] ID lists.

    predictor(context_ids, k) -> (candidate_ids, context_seen). Defaults to the
    exact-order predictor for `order`. Framing is the same for every predictor
    (max_order-1 BOS tokens), so all predictors score the identical positions.
    """
    if predictor is None:
        predictor = make_exact_predictor(model, order)
    pad = model.max_order - 1
    hits = dict.fromkeys(ks, 0)
    n = seen = suggested = skipped_unk = skipped_eos = 0
    for sent in sentences:
        padded = [BOS] * pad + sent[1:]
        for i in range(pad, len(padded)):
            target = padded[i]
            if target in SPECIAL:                 # UNK/EOS are not suggestable
                if target == UNK:
                    skipped_unk += 1
                else:
                    skipped_eos += 1
                continue
            n += 1
            cands, ctx_seen = predictor(padded[:i], max(ks))
            seen += bool(ctx_seen)
            if not cands:
                continue
            suggested += 1
            for k in ks:
                hits[k] += target in cands[:k]
    nan = float("nan")
    out = {f"top{k}": hits[k] / n if n else nan for k in ks}
    # hits only happen when a context was seen, so this division is exact
    out |= {f"top{k}_given_seen": hits[k] / seen if seen else nan for k in ks}
    out |= {"context_seen": seen / n if n else nan,
            "has_suggestion": suggested / n if n else nan,
            "evaluated": n, "skipped_unk": skipped_unk, "skipped_eos": skipped_eos}
    return out


def sentence_logprob(model, sent, order, method="backoff"):
    padded = [BOS] * (order - 1) + sent[1:]
    return sum(model.logprob(padded[i], padded[i - order + 1:i], method=method)
               for i in range(order - 1, len(padded)))


def discrimination_eval(model, sentences, order, seed=0, method="backoff"):
    rnd = random.Random(seed)
    wins = ties = n = skipped = 0
    for sent in sentences:
        body = sent[1:-1]
        if len(set(body)) < 2:
            skipped += 1
            continue
        shuffled = body[:]
        for _ in range(10):                       # avoid identical shuffles
            rnd.shuffle(shuffled)
            if shuffled != body:
                break
        else:
            skipped += 1
            continue
        a = sentence_logprob(model, sent, order, method)
        b = sentence_logprob(model, [BOS, *shuffled, EOS], order, method)
        n += 1
        if math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9):
            ties += 1
        elif a > b:
            wins += 1
    nan = float("nan")
    return {"accuracy": (wins + 0.5 * ties) / n if n else nan,   # ties = half credit
            "strict_accuracy": wins / n if n else nan,           # ties = no credit
            "wins": wins, "ties": ties, "n": n, "skipped": skipped}