from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from src.ngram_disk import DiskBackedNGramTrainer
from src.ngram_model import BOS, EOS, NGramModel


VOCAB = ["<PAD>", "<UNK>", "<s>", "</s>", "እኔ", "ቡና", "ሻይ", "እጠጣለሁ"]
SENTENCES = [
    [BOS, 4, 5, EOS],
    [BOS, 4, 5, EOS],
    [BOS, 4, 6, 7, EOS],
]
FLAT = np.asarray([token for sent in SENTENCES for token in sent], dtype=np.int32)


class DiskBackedNGramTests(unittest.TestCase):
    def _setup_paths(self, root: Path, flat: np.ndarray = FLAT):
        train, vocab = root / "train.npy", root / "vocab.txt"
        np.save(train, flat, allow_pickle=False)
        vocab.write_text("\n".join(VOCAB) + "\n", encoding="utf-8")
        return train, vocab

    def _trainer(self, root: Path, train: Path, vocab: Path, *, prefix="run", **kwargs):
        return DiskBackedNGramTrainer(
            train, vocab, root / f"{prefix}_work", root / f"{prefix}_model",
            scan_chunk_tokens=kwargs.pop("scan_chunk_tokens", 3),
            run_occurrences_limit=kwargs.pop("run_occurrences_limit", 5),
            merge_buffer_size=kwargs.pop("merge_buffer_size", 2),
            **kwargs,
        )

    def test_disk_counts_match_memory_fit_across_chunks_runs_and_duplicate_keys(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, vocab = self._setup_paths(root)
            memory = NGramModel.from_flat_ids(FLAT, VOCAB)
            disk_result = self._trainer(root, train, vocab).run()
            disk = NGramModel.load(disk_result.model_dir)
            self.assertEqual(disk_result.report["complete_sentences"], 3)
            for order in range(1, 5):
                with self.subTest(order=order):
                    self.assertEqual(disk.total_occurrences(order), memory.total_occurrences(order))
                    self.assertEqual(disk.unique_ngrams(order), memory.unique_ngrams(order))
                    np.testing.assert_array_equal(disk.orders[order].keys, memory.orders[order].keys)
                    np.testing.assert_array_equal(disk.orders[order].counts, memory.orders[order].counts)
                    np.testing.assert_array_equal(disk.orders[order].context_keys,
                                                  memory.orders[order].context_keys)
                    np.testing.assert_array_equal(disk.orders[order].context_totals,
                                                  memory.orders[order].context_totals)
            self.assertGreater(int(disk.orders[2].counts.max()), 1)
            self.assertEqual(disk.ngram_count("</s>", "<s>"), 0)
            self.assertEqual(disk.ngram_count("እኔ", "ቡና"), 2)
            self.assertEqual(disk.prob(5, [4]), memory.prob(5, [4]))
            self.assertEqual(disk.logprob(5, [4]), memory.logprob(5, [4]))
            self.assertEqual(disk.predict_next("እኔ", k=5), memory.predict_next("እኔ", k=5))
            for order, context in enumerate(([], [4], [4, 5], [2, 2, 4]), start=1):
                for method in ("mle", "add_k", "interpolation", "backoff"):
                    with self.subTest(order=order, method=method):
                        probabilities = [disk.prob(token_id, context, method=method)
                                         for token_id in disk.allowed_outcomes]
                        self.assertAlmostEqual(sum(probabilities), 1.0)
                        self.assertAlmostEqual(
                            disk.prob(5, context, method=method),
                            memory.prob(5, context, method=method),
                        )

    def test_interruption_resumes_from_committed_sentence_boundary(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, vocab = self._setup_paths(root)
            # A cap of 20 fits the first 12-occurrence sentence but not the
            # next one, forcing a flush immediately before that sentence.
            trainer = self._trainer(root, train, vocab, prefix="resume", run_occurrences_limit=20)
            original_flush = trainer._flush
            did_interrupt = False

            def interrupt_after_first_commit(batches, state, next_offset, sentence_count):
                nonlocal did_interrupt
                state = original_flush(batches, state, next_offset, sentence_count)
                if not did_interrupt and int(state["next_run_id"]) > 0:
                    did_interrupt = True
                    raise KeyboardInterrupt("simulated stop after committed run")
                return state

            trainer._flush = interrupt_after_first_commit  # type: ignore[method-assign]
            with self.assertRaises(KeyboardInterrupt):
                trainer.run()
            self.assertGreater(int(__import__("json").loads(
                trainer.checkpoint_path.read_text())["next_offset"]), 0)
            resumed = self._trainer(root, train, vocab, prefix="resume", run_occurrences_limit=20)
            result = resumed.run(resume=True)
            disk = NGramModel.load(result.model_dir)
            memory = NGramModel.from_flat_ids(FLAT, VOCAB)
            for order in range(1, 5):
                np.testing.assert_array_equal(disk.orders[order].keys, memory.orders[order].keys)
                np.testing.assert_array_equal(disk.orders[order].counts, memory.orders[order].counts)

    def test_interruption_during_merge_rebuilds_incomplete_order_without_duplicates(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, vocab = self._setup_paths(root)
            trainer = self._trainer(root, train, vocab, prefix="merge", run_occurrences_limit=5)
            original_merge = trainer._merge_order
            did_interrupt = False

            def interrupt_after_first_order(order, state):
                nonlocal did_interrupt
                merged = original_merge(order, state)
                if not did_interrupt:
                    did_interrupt = True
                    raise KeyboardInterrupt("simulated stop during merge")
                return merged

            trainer._merge_order = interrupt_after_first_order  # type: ignore[method-assign]
            with self.assertRaises(KeyboardInterrupt):
                trainer.run()
            resumed = self._trainer(root, train, vocab, prefix="merge", run_occurrences_limit=5)
            result = resumed.run(resume=True)
            disk = NGramModel.load(result.model_dir)
            memory = NGramModel.from_flat_ids(FLAT, VOCAB)
            for order in range(1, 5):
                np.testing.assert_array_equal(disk.orders[order].keys, memory.orders[order].keys)
                np.testing.assert_array_equal(disk.orders[order].counts, memory.orders[order].counts)

    def test_max_tokens_uses_only_complete_sentence_prefix(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, vocab = self._setup_paths(root)
            trainer = self._trainer(root, train, vocab, prefix="prefix", max_tokens=6)
            result = trainer.run()
            model = NGramModel.load(result.model_dir)
            self.assertEqual(result.report["complete_sentences"], 1)
            self.assertEqual(result.report["source_ids_processed"], 4)
            self.assertEqual(model.total_occurrences(1), 3)

    def test_bad_sentence_boundaries_are_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, vocab = self._setup_paths(root, np.asarray([BOS, 4, EOS, EOS], dtype=np.int32))
            trainer = self._trainer(root, train, vocab, prefix="bad")
            with self.assertRaisesRegex(ValueError, "outside sentence"):
                trainer.run()


if __name__ == "__main__":
    unittest.main()
