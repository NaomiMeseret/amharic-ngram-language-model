import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.preprocess_lib import tokenize
from src.ngram_model import (
    BOS,
    EOS,
    NGramModel,
    iter_encoded_sentences,
    normalize_and_tokenize,
)


CORPUS = [
    ["እኔ", "ቡና", "እወዳለሁ"],
    ["እኔ", "ቡና", "እጠጣለሁ"],
    ["እኔ", "ሻይ", "እጠጣለሁ"],
]
VOCAB = ["<PAD>", "<UNK>", "<s>", "</s>", "እኔ", "ቡና", "እወዳለሁ", "እጠጣለሁ", "ሻይ"]
W2I = {word: i for i, word in enumerate(VOCAB)}
FRAMED = [[BOS, *(W2I[word] for word in sentence), EOS] for sentence in CORPUS]


class NGramModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = NGramModel.fit(FRAMED, VOCAB, max_order=4)

    def test_manual_occurrence_and_unique_counts_for_all_orders(self):
        # Nine ordinary words plus one EOS prediction in each of three sentences.
        for order in range(1, 5):
            with self.subTest(order=order):
                self.assertEqual(self.model.total_occurrences(order), 12)
                self.assertGreater(self.model.unique_ngrams(order), 0)
        self.assertEqual(self.model.unique_ngrams(2), 8)
        self.assertEqual(self.model.ngram_count("እኔ", "ቡና"), 2)
        self.assertEqual(self.model.ngram_count("<s>", "<s>", "እኔ"), 3)
        self.assertEqual(self.model.ngram_count("<s>", "<s>", "<s>", "እኔ"), 3)
        self.assertEqual(self.model.ngram_count("እኔ", "ቡና", "እወዳለሁ", "</s>"), 1)

    def test_n_grams_never_cross_sentence_boundaries(self):
        for order in range(2, 5):
            with self.subTest(order=order):
                self.assertEqual(self.model.ngram_count("</s>", "<s>"), 0)
                self.assertEqual(self.model.ngram_count_ids([EOS, BOS] + [BOS] * (order - 2)), 0)
                # Virtual BOS contexts provide each sentence-initial word one
                # full prediction context without making an EOS->BOS transition.
                self.assertEqual(self.model.total_occurrences(order), 12)

    def test_flat_stream_decoder_and_malformed_boundaries(self):
        flattened = [token for sentence in FRAMED for token in sentence]
        self.assertEqual(list(iter_encoded_sentences(flattened)), FRAMED)
        for bad_stream in ([BOS, 4, 4], [EOS], [4, BOS, 4, EOS], [BOS, 0, EOS]):
            with self.subTest(stream=bad_stream), self.assertRaises(ValueError):
                list(iter_encoded_sentences(bad_stream))

    def test_mle_probability_and_logprob(self):
        self.assertAlmostEqual(self.model.prob(W2I["ቡና"], [W2I["እኔ"]]), 2 / 3)
        self.assertAlmostEqual(self.model.prob(W2I["ሻይ"], [W2I["እኔ"]]), 1 / 3)
        self.assertAlmostEqual(self.model.logprob(W2I["ቡና"], [W2I["እኔ"]]), math.log(2 / 3))
        self.assertEqual(self.model.prob(W2I["እኔ"], [W2I["ቡና"], W2I["ሻይ"]]), 0.0)
        self.assertEqual(self.model.logprob(W2I["እኔ"], [W2I["ቡና"], W2I["ሻይ"]]), -math.inf)
        self.assertEqual(self.model.prob(0, [W2I["እኔ"]]), 0.0)  # PAD is not a prediction outcome
        self.assertEqual(self.model.prob(BOS, [W2I["እኔ"]]), 0.0)
        self.assertAlmostEqual(sum(self.model.conditional_distribution(["እኔ"]).values()), 1.0)

    def test_add_k_smoothing_uses_prediction_vocabulary(self):
        # V=7: UNK, EOS, and the five ordinary words; PAD and BOS are excluded.
        self.assertEqual(len(self.model.allowed_outcomes), 7)
        self.assertAlmostEqual(self.model.prob(W2I["ቡና"], [W2I["እኔ"]], method="add_k", k=1), 3 / 10)
        self.assertAlmostEqual(self.model.prob(W2I["እኔ"], [W2I["ቡና"], W2I["ሻይ"]],
                                              method="add_k", k=1), 1 / 7)
        distribution = self.model.conditional_distribution(["እኔ"], smoothing=1)
        self.assertEqual(len(distribution), 7)
        self.assertAlmostEqual(sum(distribution.values()), 1.0)
        with self.assertRaises(ValueError):
            self.model.prob(W2I["ቡና"], [W2I["እኔ"]], method="add_k", k=-1)

    def test_add_k_zero_matches_mle_and_handles_unseen_context(self):
        seen_context = [W2I["እኔ"]]
        unseen_context = [W2I["ቡና"], W2I["ሻይ"]]
        target = W2I["እጠጣለሁ"]

        self.assertAlmostEqual(
            self.model.prob(target, seen_context, method="add_k", k=0),
            self.model.prob(target, seen_context, method="mle"),
        )
        self.assertEqual(
            self.model.prob(target, unseen_context, method="add_k", k=0), 0.0
        )
        self.assertEqual(
            self.model.logprob(target, unseen_context, method="add_k", k=0),
            -math.inf,
        )

    def test_all_probability_methods_normalize_at_each_order(self):
        contexts = [
            [],
            [W2I["እኔ"]],
            [W2I["እኔ"], W2I["ቡና"]],
            [BOS, BOS, BOS],
        ]
        for order, context in enumerate(contexts, start=1):
            with self.subTest(order=order):
                for method in ("mle", "add_k", "interpolation", "backoff"):
                    kwargs = {"k": 1} if method == "add_k" else {}
                    total = sum(self.model.prob(token_id, context, method=method, **kwargs)
                                for token_id in self.model.allowed_outcomes)
                    self.assertAlmostEqual(total, 1.0, msg=f"{method} order {order}")

    def test_linear_interpolation_is_normalized_and_weighted(self):
        context = [W2I["እኔ"]]
        value = self.model.prob(W2I["ቡና"], context, method="interpolation", weights=[0.25, 0.75])
        expected = 0.25 * (2 / 12) + 0.75 * (2 / 3)
        self.assertAlmostEqual(value, expected)
        distribution = self.model.conditional_distribution(["ቡና", "ሻይ"], method="interpolation")
        self.assertAlmostEqual(sum(distribution.values()), 1.0)
        with self.assertRaises(ValueError):
            self.model.prob(W2I["ቡና"], context, method="interpolation", weights=[0.2, 0.2])

    def test_backoff_uses_longest_seen_context(self):
        # The trigram context (ቡና, ሻይ) is unseen. The bigram suffix (ሻይ)
        # occurs once and is followed by እጠጣለሁ.
        self.assertEqual(self.model.prob(W2I["እጠጣለሁ"], [W2I["ቡና"], W2I["ሻይ"]]), 0.0)
        self.assertEqual(self.model.prob(W2I["እጠጣለሁ"], [W2I["ቡና"], W2I["ሻይ"]],
                                         method="backoff"), 0.5)
        self.assertEqual(self.model.prob(W2I["እጠጣለሁ"], [W2I["እኔ"], W2I["ቡና"]],
                                         method="backoff"), 0.25)
        distribution = self.model.conditional_distribution(["እኔ"], method="backoff")
        self.assertAlmostEqual(sum(distribution.values()), 1.0)

    def test_prediction_ranking_and_probability_scores(self):
        predictions = self.model.predict_next("እኔ", k=5)
        self.assertEqual(predictions[:2], [("ቡና", 0.5), ("ሻይ", 1 / 6)])
        self.assertNotIn("</s>", dict(predictions))
        self.assertNotIn("<UNK>", dict(predictions))
        unigram_predictions = self.model.predict_next([], k=5)
        # Scores retain their full-event denominators (EOS/UNK are not renormalized away).
        self.assertAlmostEqual(sum(score for _, score in unigram_predictions), 9 / 12)
        self.assertEqual(self.model.predict_next("", k=0), [])
        self.assertEqual(self.model.predict_next("", k=5), sorted(
            self.model.predict_next("", k=5), key=lambda item: (-item[1], item[0])))

    def test_preprocessing_tokenizer_is_shared_with_prediction(self):
        for text in ("ሐምሌ 19ኛ", "በ1980 ዓ.ም", "2010 2011", "10/12/2010"):
            with self.subTest(text=text):
                self.assertEqual(normalize_and_tokenize(text), tokenize(text))

    def test_fit_rejects_invalid_vocabulary_and_ids(self):
        with self.assertRaises(ValueError):
            NGramModel.fit(FRAMED, ["<UNK>", "<PAD>", "<s>", "</s>"])
        with self.assertRaises(ValueError):
            NGramModel.fit([[BOS, 999, EOS]], VOCAB)
        with self.assertRaises(ValueError):
            NGramModel.fit([[BOS, 0, EOS]], VOCAB)
        with self.assertRaises(ValueError):
            NGramModel.fit([[BOS, 4]], VOCAB)
        with self.assertRaises(MemoryError):
            NGramModel.fit(FRAMED, VOCAB, max_occurrences=47)

    def test_save_and_reload_sparse_model(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model"
            self.model.save(path)
            loaded = NGramModel.load(path)
            self.assertEqual(loaded.total_occurrences(4), self.model.total_occurrences(4))
            self.assertEqual(loaded.ngram_count("እኔ", "ቡና"), 2)
            self.assertAlmostEqual(loaded.prob(W2I["ቡና"], [W2I["እኔ"]]), 2 / 3)
            self.assertAlmostEqual(loaded.logprob(W2I["ቡና"], [W2I["እኔ"]]), math.log(2 / 3))
            self.assertEqual(loaded.predict_next("እኔ", k=5), self.model.predict_next("እኔ", k=5))
            with self.assertRaises(FileExistsError):
                self.model.save(path)


if __name__ == "__main__":
    unittest.main()
