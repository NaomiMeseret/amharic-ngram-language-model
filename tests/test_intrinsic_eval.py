import unittest
import numpy as np
from src.ngram_model import BOS, EOS
from src.intrinsic_eval import sample_sentences


class SampleSentencesBoundaryTests(unittest.TestCase):
    def test_valid_stream_gives_complete_sentences(self):
        out = sample_sentences(np.array([BOS, 10, 11, EOS, BOS, 12, EOS]), 1, seed=1)
        self.assertEqual(out[0], BOS)
        self.assertEqual(out[-1], EOS)

    def test_nested_start_rejected(self):
        with self.assertRaises(ValueError):
            sample_sentences(np.array([BOS, 10, BOS, 11, EOS, EOS]), 1)

    def test_end_without_open_sentence_rejected(self):
        with self.assertRaises(ValueError):
            sample_sentences(np.array([BOS, 10, EOS, EOS, BOS, 11, EOS]), 1)

    def test_missing_final_end_rejected(self):
        with self.assertRaises(ValueError):
            sample_sentences(np.array([BOS, 10, EOS, BOS, 11]), 1)