"""Tests for src/extrinsic_eval.py (next-word prediction and discrimination)."""
import math
import unittest

import numpy as np

import src.extrinsic_eval as ext
from src.ngram_model import BOS, EOS, UNK


class FakeOrder:
    """One n-gram order: next_counts(context) -> [(token_id, count), ...]."""
    def __init__(self, table):
        self.table = table

    def next_counts(self, ctx):
        return self.table.get(tuple(ctx), [])


class FakeUnigrams:
    """Order-1 storage: parallel arrays of token ids and counts."""
    def __init__(self, counts_by_token):
        tokens = list(counts_by_token)
        self.keys = np.array(tokens)
        self.counts = np.array([counts_by_token[t] for t in tokens])


class FakeModel:
    def __init__(self, max_order=2, tables=None, unigrams=None, lp=None):
        self.max_order = max_order
        self.orders = {n: FakeOrder(t) for n, t in (tables or {}).items()}
        if unigrams:
            self.orders[1] = FakeUnigrams(unigrams)
        self._lp = lp or (lambda tok, ctx: -1.0)

    def logprob(self, tok, ctx, method="backoff"):
        return self._lp(tok, tuple(ctx))


class DiscriminationTests(unittest.TestCase):
    def test_tie_gets_half_credit(self):
        r = ext.discrimination_eval(FakeModel(), [[BOS, 10, 11, 12, EOS]], order=2)
        self.assertEqual((r["ties"], r["wins"], r["n"]), (1, 0, 1))
        self.assertEqual(r["accuracy"], 0.5)
        self.assertEqual(r["strict_accuracy"], 0.0)

    def test_real_beats_shuffled(self):
        real = [BOS, 10, 11, 12, EOS]
        good = set(zip(real[:-1], real[1:]))                # (previous, token) pairs of the real order
        lp = lambda tok, ctx: -1.0 if (ctx[-1], tok) in good else -5.0
        r = ext.discrimination_eval(FakeModel(lp=lp), [real], order=2)
        self.assertEqual((r["wins"], r["ties"], r["n"]), (1, 0, 1))
        self.assertEqual(r["accuracy"], 1.0)

    def test_unshufflable_sentence_is_skipped(self):
        r = ext.discrimination_eval(FakeModel(), [[BOS, 10, 10, EOS]], order=2)
        self.assertEqual((r["n"], r["skipped"]), (0, 1))
        self.assertTrue(math.isnan(r["accuracy"]))


class NextWordTests(unittest.TestCase):
    def test_unk_and_eos_skipped_and_counted_separately(self):
        m = FakeModel(tables={2: {(BOS,): [(10, 5)]}})
        r = ext.next_word_eval(m, [[BOS, 10, UNK, EOS]], order=2, ks=(1,))
        self.assertEqual((r["evaluated"], r["skipped_unk"], r["skipped_eos"]), (1, 1, 1))
        self.assertEqual(r["top1"], 1.0)

    def test_context_seen_differs_from_has_suggestion(self):
        # context (10,) was seen, but only ever followed by </s>
        m = FakeModel(tables={2: {(BOS,): [(10, 5)], (10,): [(EOS, 3)]}})
        r = ext.next_word_eval(m, [[BOS, 10, 11, EOS]], order=2, ks=(1,))
        self.assertEqual(r["context_seen"], 1.0)
        self.assertEqual(r["has_suggestion"], 0.5)
        self.assertEqual(r["top1"], 0.5)

    def test_unseen_context_counts_as_miss(self):
        m = FakeModel(tables={2: {}})
        r = ext.next_word_eval(m, [[BOS, 10, EOS]], order=2, ks=(1,))
        self.assertEqual((r["evaluated"], r["top1"], r["context_seen"]), (1, 0.0, 0.0))

    def test_topk_ranking_by_count(self):
        m = FakeModel(tables={2: {(BOS,): [(10, 1), (11, 5), (12, 3)]}})   # ranking: 11, 12, 10
        r = ext.next_word_eval(m, [[BOS, 12, EOS]], order=2, ks=(1, 3))
        self.assertEqual(r["top1"], 0.0)
        self.assertEqual(r["top3"], 1.0)

    def test_unigram_ranking_excludes_special_tokens(self):
        m = FakeModel(max_order=1, unigrams={10: 5, 11: 9, UNK: 100})      # ranking: 11, 10
        r = ext.next_word_eval(m, [[BOS, 10, EOS]], order=1, ks=(1, 3))
        self.assertEqual(r["top1"], 0.0)
        self.assertEqual(r["top3"], 1.0)

    def test_sentence_start_uses_bos_padding(self):
        m = FakeModel(max_order=3, tables={3: {(BOS, BOS): [(10, 4)], (BOS, 10): [(11, 2)]}})
        r = ext.next_word_eval(m, [[BOS, 10, 11, EOS]], order=3, ks=(1,))
        self.assertEqual((r["evaluated"], r["top1"]), (2, 1.0))

    def test_same_positions_for_every_order(self):
        m = FakeModel(max_order=3, unigrams={10: 5, 11: 3},
                      tables={2: {(BOS,): [(10, 2)], (10,): [(11, 2)]},
                              3: {(BOS, BOS): [(10, 2)], (BOS, 10): [(11, 2)]}})
        sents = [[BOS, 10, 11, EOS]]
        results = [ext.next_word_eval(m, sents, order=n) for n in (1, 2, 3)]
        self.assertEqual({r["evaluated"] for r in results}, {2})
        self.assertEqual({r["skipped_eos"] for r in results}, {1})


if __name__ == "__main__":
    unittest.main()