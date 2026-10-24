import unittest

import numpy as np

from src.ngram_model import BOS, EOS
from src.train_ngram_slice import select_complete_prefix


class SliceSelectionTests(unittest.TestCase):
    def test_selects_last_complete_sentence_within_limit(self):
        stream = np.array([BOS, 4, EOS, BOS, 5, 6, EOS, BOS, 7, EOS], dtype=np.int32)
        selected, sentences, inspected = select_complete_prefix(stream, 8, scan_chunk_size=2)
        self.assertEqual(selected.tolist(), stream[:7].tolist())
        self.assertEqual(sentences, 2)
        self.assertEqual(inspected, 8)
        self.assertLessEqual(selected.size, 8)

    def test_drops_unfinished_trailing_sentence(self):
        stream = np.array([BOS, 4, EOS, BOS, 5, 6], dtype=np.int32)
        selected, sentences, inspected = select_complete_prefix(stream, 6)
        self.assertEqual(selected.tolist(), stream[:3].tolist())
        self.assertEqual(sentences, 1)
        self.assertEqual(inspected, 6)

    def test_no_complete_sentence_returns_empty_view(self):
        stream = np.array([BOS, 4, 5], dtype=np.int32)
        selected, sentences, inspected = select_complete_prefix(stream, 3)
        self.assertEqual(selected.size, 0)
        self.assertEqual(sentences, 0)
        self.assertEqual(inspected, 3)

    def test_rejects_invalid_limits(self):
        stream = np.array([], dtype=np.int32)
        with self.assertRaises(ValueError):
            select_complete_prefix(stream, 0)


if __name__ == "__main__":
    unittest.main()
