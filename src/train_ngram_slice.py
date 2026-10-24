"""Build and validate four sparse N-gram orders on a sentence-safe train slice.

Example (run from the repository root):
  python3 -m src.train_ngram_slice --max-tokens 2000000 \
    --output-dir results/ngram_slices/train_2m

The input is memory-mapped. Only the requested prefix is scanned, and the
selected view ends at the last complete </s> within that prefix.
"""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import islice
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

from src.ngram_model import BOS, EOS, NGramModel, iter_encoded_sentences


def select_complete_prefix(array: np.ndarray, requested_limit: int,
                           eos_id: int = EOS, scan_chunk_size: int = 1_000_000
                           ) -> tuple[np.ndarray, int, int]:
    """Return a memmap view through the last EOS within the first N IDs.

    Returns ``(selected_view, complete_sentence_count, inspected_id_count)``.
    ``inspected_id_count`` is at most the requested limit. If no EOS occurs in
    that prefix, the returned view is empty and sentence count is zero.
    """
    if requested_limit < 1:
        raise ValueError("--max-tokens must be a positive integer")
    if scan_chunk_size < 1:
        raise ValueError("scan_chunk_size must be a positive integer")
    inspected = min(requested_limit, int(array.size))
    last_eos = -1
    for start in range(0, inspected, scan_chunk_size):
        stop = min(start + scan_chunk_size, inspected)
        local = np.flatnonzero(array[start:stop] == eos_id)
        if local.size:
            last_eos = start + int(local[-1])
    if last_eos < 0:
        return array[:0], 0, inspected
    end = last_eos + 1
    selected = array[:end]
    sentence_count = int(np.count_nonzero(selected == BOS))
    return selected, sentence_count, inspected


def _check_ngram_structure(model: NGramModel, sentence_count: int) -> dict[str, object]:
    checks: dict[str, object] = {"valid": True, "failure_messages": []}
    failures: list[str] = checks["failure_messages"]  # type: ignore[assignment]
    if model.ngram_count("</s>", "<s>") != 0:
        failures.append("EOS->BOS bigram exists")
    eos_ending = {}
    early_eos = {}
    bos_targets = {}
    for order, counts in model.orders.items():
        key_ids = [((counts.keys >> np.uint64(16 * shift)) & np.uint64(0xFFFF))
                   for shift in range(order - 1, -1, -1)]
        if order > 1:
            early = np.zeros(len(counts.keys), dtype=bool)
            for ids in key_ids[:-1]:
                early |= ids == EOS
            early_eos[order] = int(np.count_nonzero(early))
            if early_eos[order]:
                failures.append(f"order {order} has EOS before its final position")
        endings = key_ids[-1] == EOS
        eos_ending[order] = int(counts.counts[endings].sum(dtype=np.uint64))
        if eos_ending[order] != sentence_count:
            failures.append(f"order {order} has {eos_ending[order]} EOS targets, expected {sentence_count}")
        bos_targets[order] = int(counts.counts[key_ids[-1] == BOS].sum(dtype=np.uint64))
        if bos_targets[order]:
            failures.append(f"order {order} predicts <s>")
    checks.update({"eos_to_bos_bigram_count": model.ngram_count("</s>", "<s>"),
                   "ngram_counts_ending_eos": eos_ending,
                   "ngram_types_with_eos_early": early_eos,
                   "predicted_bos_occurrences": bos_targets,
                   "valid": not failures})
    return checks


def _check_manual_counts(model: NGramModel, selected: np.ndarray) -> dict[str, object]:
    """Independently verify unigram and sentence-initial counts from the slice."""
    histogram = np.bincount(selected, minlength=len(model.vocabulary))
    if len(histogram) < len(model.vocabulary):
        raise AssertionError("histogram was shorter than the vocabulary")
    for token_id, expected in enumerate(histogram):
        if token_id in {0, BOS}:  # PAD is forbidden; BOS is context only.
            expected = 0
        if model.ngram_count_ids([token_id]) != int(expected):
            raise AssertionError(f"unigram count mismatch for token ID {token_id}")

    first_targets: Counter[int] = Counter()
    for sentence in iter_encoded_sentences(selected):
        if len(sentence) < 2:
            raise AssertionError("decoded sentence is missing its EOS")
        first_targets[sentence[1]] += 1
    checked_prefix_types = 0
    for token_id, count in first_targets.items():
        for order in range(2, 5):
            key = [BOS] * (order - 1) + [token_id]
            if model.ngram_count_ids(key) != count:
                raise AssertionError(f"order-{order} sentence-initial count mismatch for ID {token_id}")
            checked_prefix_types += 1

    # Independently count every N-gram in a small decoded subset and verify
    # it is included in the full slice's count.
    subset_counts: dict[int, Counter[tuple[int, ...]]] = {
        order: Counter() for order in range(1, 5)
    }
    subset_sentences = list(islice(iter_encoded_sentences(selected), 100))
    for sentence in subset_sentences:
        words_and_eos = sentence[1:]
        for order in range(1, 5):
            tokens = ([BOS] * (order - 1) + words_and_eos) if order > 1 else words_and_eos
            for index in range(len(tokens) - order + 1):
                subset_counts[order][tuple(tokens[index:index + order])] += 1
    checked_subset_types = 0
    for order, counter in subset_counts.items():
        for ngram, count in counter.items():
            if model.ngram_count_ids(ngram) < count:
                raise AssertionError(f"order-{order} model count below manual subset count for {ngram}")
            checked_subset_types += 1
    return {
        "all_unigram_counts_match_numpy_histogram": True,
        "sentence_initial_context_counts_match": True,
        "sentence_initial_token_types_checked": len(first_targets),
        "prefix_ngram_checks": checked_prefix_types,
        "manual_subset_sentences": len(subset_sentences),
        "manual_subset_ngram_types_checked": checked_subset_types,
    }


def _check_probability_methods(model: NGramModel) -> dict[str, object]:
    contexts = [[], [BOS], [BOS, BOS], [BOS, BOS, BOS]]
    add_k_sums = {}
    interpolation_sums = {}
    for order, context in enumerate(contexts, start=1):
        add_k_sums[str(order)] = float(sum(
            model.prob(word_id, context, method="add_k", k=1.0)
            for word_id in model.allowed_outcomes
        ))
        interpolation_sums[str(order)] = float(sum(
            model.prob(word_id, context, method="interpolation")
            for word_id in model.allowed_outcomes
        ))

    observed_context = [BOS]
    observed = model.orders[2].next_counts(observed_context)
    mle_observed_sum = sum(model.prob(word_id, observed_context, method="mle")
                           for word_id, _ in observed)
    if not np.isclose(mle_observed_sum, 1.0, atol=1e-6):
        raise AssertionError(f"MLE probabilities over observed continuations sum to {mle_observed_sum}")

    target = next((word_id for word_id in model.allowed_outcomes
                   if model.prob(word_id, [BOS], method="add_k", k=1) > 0), None)
    if target is None:
        raise AssertionError("no prediction target found")
    p = model.prob(target, [BOS], method="add_k", k=1)
    lp = model.logprob(target, [BOS], method="add_k", k=1)
    if not np.isclose(lp, np.log(p), atol=1e-12):
        raise AssertionError("logprob does not equal log(prob)")

    # Find an actually unseen four-gram event for which smoothing/backoff apply.
    last_id = len(model.vocabulary) - 1
    unseen_context = [BOS, BOS, last_id]
    unseen_word = next((word_id for word_id in model.allowed_outcomes
                        if model.ngram_count_ids([*unseen_context, word_id]) == 0), None)
    if unseen_word is None:
        unseen_context = [last_id, last_id, last_id]
        unseen_word = next((word_id for word_id in model.allowed_outcomes
                            if model.ngram_count_ids([*unseen_context, word_id]) == 0), None)
    if unseen_word is None:
        raise AssertionError("could not find an unseen four-gram event")
    unseen_add_k = model.prob(unseen_word, unseen_context, method="add_k", k=1)
    unseen_backoff = model.prob(unseen_word, unseen_context, method="backoff")

    sums_ok = all(np.isclose(value, 1.0, atol=1e-6)
                  for value in (*add_k_sums.values(), *interpolation_sums.values()))
    if not sums_ok or unseen_add_k <= 0 or unseen_backoff < 0:
        raise AssertionError("probability normalization or unseen-event check failed")
    return {
        "add_k_distribution_sums_by_order": add_k_sums,
        "interpolation_distribution_sums_by_order": interpolation_sums,
        "mle_sum_over_observed_bigram_continuations": float(mle_observed_sum),
        "logprob_check": {"probability": float(p), "logprob": float(lp),
                          "natural_log_probability": float(np.log(p))},
        "unseen_fourgram_event": {"context_ids": unseen_context, "word_id": int(unseen_word),
                                   "add_k_probability": float(unseen_add_k),
                                   "backoff_probability": float(unseen_backoff)},
    }


def _validate_model_reload(model: NGramModel, model_dir: Path) -> dict[str, object]:
    loaded = NGramModel.load(model_dir, mmap_mode="r")
    contexts = [[], [BOS], [BOS, BOS], [BOS, BOS, BOS]]
    target_id = loaded.word_to_id.get("<UNK>", 1)
    for context in contexts:
        for method in ("mle", "add_k", "interpolation", "backoff"):
            before = model.prob(target_id, context, method=method)
            after = loaded.prob(target_id, context, method=method)
            if not np.isclose(before, after, atol=1e-12):
                raise AssertionError("probability changed after reload")
        before_log = model.logprob(target_id, context, method="add_k")
        after_log = loaded.logprob(target_id, context, method="add_k")
        if not np.isclose(before_log, after_log, atol=1e-12):
            raise AssertionError("logprob changed after reload")
    predictions_before = model.predict_next("ሐምሌ absentword", k=5)
    predictions_after = loaded.predict_next("ሐምሌ absentword", k=5)
    if predictions_before != predictions_after:
        raise AssertionError("predictions changed after reload")
    long_context = [BOS, BOS, BOS, BOS, target_id]
    clipped_context = long_context[-3:]
    p_long = loaded.prob(target_id, long_context, method="add_k")
    p_clipped = loaded.prob(target_id, clipped_context, method="add_k")
    if not np.isclose(p_long, p_clipped, atol=1e-12):
        raise AssertionError("contexts longer than the supported history were not clipped")
    return {
        "loaded_with_mmap": True,
        "prob_and_logprob_match": True,
        "prediction_match": True,
        "normalized_unknown_context": True,
        "long_context_clipped_to_final_3_ids": True,
        "sample_predictions": predictions_after,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    train_path = Path(args.train_path)
    vocab_path = Path(args.vocab_path)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists; refusing to overwrite: {output_dir}")
    vocab = vocab_path.read_text(encoding="utf-8").splitlines()
    if vocab[:4] != ["<PAD>", "<UNK>", "<s>", "</s>"]:
        raise ValueError("vocabulary must begin with <PAD>, <UNK>, <s>, </s>")
    train = np.load(train_path, mmap_mode="r", allow_pickle=False)
    if train.ndim != 1 or train.dtype != np.int32:
        raise ValueError(f"expected a flattened int32 train array, got {train.shape} {train.dtype}")
    selected, sentence_count, inspected = select_complete_prefix(train, args.max_tokens)
    if selected.size == 0:
        return {"status": "no_complete_sentence", "requested_token_limit": args.max_tokens,
                "inspected_token_ids": inspected, "actual_token_ids_processed": 0,
                "complete_sentences": 0, "output_created": False}
    min_id, max_id = int(selected.min()), int(selected.max())
    if min_id < 0 or max_id >= len(vocab):
        raise ValueError(f"selected input has vocabulary ID range [{min_id}, {max_id}] "
                         f"outside vocab size {len(vocab)}")
    if sentence_count != int(np.count_nonzero(selected == EOS)):
        raise ValueError("selected prefix has unpaired start/end marker counts")

    count_start = time.perf_counter()
    model = NGramModel.from_flat_ids(selected, vocab, max_order=4,
                                     max_occurrences=args.max_occurrences)
    counting_seconds = time.perf_counter() - count_start
    structure_checks = _check_ngram_structure(model, sentence_count)
    if not structure_checks["valid"]:
        raise AssertionError(structure_checks["failure_messages"])
    probability_checks = _check_probability_methods(model)
    manual_count_checks = _check_manual_counts(model, selected)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    save_start = time.perf_counter()
    model_dir = model.save(output_dir / "model")
    model_save_seconds = time.perf_counter() - save_start
    reload_start = time.perf_counter()
    reload_checks = _validate_model_reload(model, model_dir)
    reload_validation_seconds = time.perf_counter() - reload_start
    artifact_files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    size_by_file = {str(path.relative_to(output_dir)): path.stat().st_size for path in artifact_files}
    peak_rss_raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    rss_unit = "bytes" if sys.platform == "darwin" else "KiB"
    peak_rss_bytes = peak_rss_raw if rss_unit == "bytes" else peak_rss_raw * 1024
    report: dict[str, object] = {
        "status": "ok",
        "train_path": str(train_path),
        "vocab_path": str(vocab_path),
        "output_dir": str(output_dir),
        "requested_token_limit": args.max_tokens,
        "inspected_token_ids": inspected,
        "actual_token_ids_processed": int(selected.size),
        "complete_sentences": sentence_count,
        "token_id_range": [min_id, max_id],
        "counting_seconds": counting_seconds,
        "model_save_seconds": model_save_seconds,
        "reload_validation_seconds": reload_validation_seconds,
        "save_and_reload_seconds": model_save_seconds + reload_validation_seconds,
        "max_occurrences_guard": args.max_occurrences,
        "model_counts": {
            str(order): {"total_occurrences": model.total_occurrences(order),
                         "unique_ngrams": model.unique_ngrams(order)}
            for order in range(1, 5)
        },
        "structure_checks": structure_checks,
        "manual_count_checks": manual_count_checks,
        "probability_checks": probability_checks,
        "reload_checks": reload_checks,
        "predict_next_full_distribution_note": (
            "Displayed top-5 suggestions are a subset. Their scores remain probabilities "
            "from the full prediction distribution; they are not renormalized over top-5."),
        "artifact_sizes_bytes": size_by_file,
        "total_artifact_bytes": sum(size_by_file.values()),
        "temporary_disk_usage_bytes": 0,
        "peak_rss_raw": peak_rss_raw,
        "peak_rss_raw_unit": rss_unit,
        "peak_rss_bytes": peak_rss_bytes,
        "peak_rss_mib": round(peak_rss_bytes / (1024 ** 2), 2),
    }
    report_path = output_dir / "run_report.json"
    report["artifact_sizes_bytes"]["run_report.json"] = 0  # type: ignore[index]
    for _ in range(4):
        report["total_artifact_bytes"] = sum(report["artifact_sizes_bytes"].values())  # type: ignore[union-attr]
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        size = report_path.stat().st_size
        if report["artifact_sizes_bytes"]["run_report.json"] == size:  # type: ignore[index]
            break
        report["artifact_sizes_bytes"]["run_report.json"] = size  # type: ignore[index]
    report["total_artifact_bytes"] = sum(report["artifact_sizes_bytes"].values())  # type: ignore[union-attr]
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default="data/processed/train.npy")
    parser.add_argument("--vocab-path", default="data/vocab.txt")
    parser.add_argument("--output-dir", required=True,
                        help="new output directory; existing paths are never overwritten")
    parser.add_argument("--max-tokens", type=int, required=True,
                        help="inspect at most this many leading train token IDs")
    parser.add_argument("--max-occurrences", type=int, default=10_000_000,
                        help="hard guard on raw packed keys across orders")
    args = parser.parse_args()
    try:
        report = run(args)
    except Exception as exc:
        parser.exit(2, f"slice run failed: {type(exc).__name__}: {exc}\n")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
