"""Bounded-memory external counting for sentence-framed flattened N-gram data.

The input is a one-dimensional ``.npy`` stream of vocabulary IDs. Intermediate
runs are sorted ``.npy`` key/count pairs; final tables are raw little-endian
uint64 files that :class:`src.ngram_model.NGramModel` memory-maps on load.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import json
import os
from pathlib import Path
import shutil
import time
from typing import Sequence

import numpy as np

from src.ngram_model import BOS, EOS, PAD, NGramModel, _pack_windows
from src.preprocess_lib import RULES_VERSION


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _pack_sentence(sentence: Sequence[int], order: int) -> np.ndarray:
    """Pack word/EOS targets with virtual BOS context padding."""
    targets = np.asarray(sentence[1:], dtype=np.uint16)
    if order == 1:
        return targets.astype(np.uint64)
    framed = np.concatenate((np.full(order - 1, BOS, dtype=np.uint16), targets))
    return _pack_windows(framed, order)


@dataclass
class DiskRunResult:
    model_dir: Path
    manifest_path: Path
    report: dict[str, object]


class DiskBackedNGramTrainer:
    """Count n-grams using bounded local batches and external sorted runs.

    ``run_occurrences_limit`` caps combined raw keys retained in NumPy arrays
    before writing sorted runs. A single sentence is kept intact, so the cap
    may be exceeded by at most that sentence's (bounded) contribution.
    Checkpoints advance only after complete sentences and committed runs.
    """

    def __init__(self, train_path: str | Path, vocab_path: str | Path,
                 work_dir: str | Path, output_dir: str | Path, *,
                 max_order: int = 4, scan_chunk_tokens: int = 1_000_000,
                 run_occurrences_limit: int = 5_000_000,
                 merge_buffer_size: int = 262_144,
                 max_tokens: int | None = None,
                 max_sentence_tokens: int | None = 100):
        self.train_path = Path(train_path)
        self.vocab_path = Path(vocab_path)
        self.work_dir = Path(work_dir)
        self.output_dir = Path(output_dir)
        self.max_order = max_order
        self.scan_chunk_tokens = scan_chunk_tokens
        self.run_occurrences_limit = run_occurrences_limit
        self.merge_buffer_size = merge_buffer_size
        self.max_tokens = max_tokens
        self.max_sentence_tokens = max_sentence_tokens
        if max_order < 1 or max_order > 4:
            raise ValueError("max_order must be between one and four")
        if min(scan_chunk_tokens, run_occurrences_limit, merge_buffer_size) < 1:
            raise ValueError("chunk, run, and merge buffer sizes must be positive")
        if max_tokens is not None and max_tokens < 1:
            raise ValueError("max_tokens must be positive when provided")
        self.vocabulary = self.vocab_path.read_text(encoding="utf-8").splitlines()
        if (len(self.vocabulary) < 4 or self.vocabulary[:4] !=
                ["<PAD>", "<UNK>", "<s>", "</s>"]):
            raise ValueError("vocabulary must begin with <PAD>, <UNK>, <s>, </s>")
        if len(self.vocabulary) >= 65536 or len(set(self.vocabulary)) != len(self.vocabulary):
            raise ValueError("vocabulary must be unique and smaller than 65,536 entries")
        self._signature = self._input_signature()

    def _input_signature(self) -> dict[str, object]:
        stat = self.train_path.stat()
        vocab_hash = hashlib.sha256(self.vocab_path.read_bytes()).hexdigest()
        return {
            "train_path": str(self.train_path.resolve()),
            "train_size_bytes": stat.st_size,
            "train_mtime_ns": stat.st_mtime_ns,
            "vocabulary_sha256": vocab_hash,
            "max_order": self.max_order,
            "scan_chunk_tokens": self.scan_chunk_tokens,
            "run_occurrences_limit": self.run_occurrences_limit,
            "merge_buffer_size": self.merge_buffer_size,
            "max_tokens": self.max_tokens,
            "max_sentence_tokens": self.max_sentence_tokens,
        }

    @property
    def checkpoint_path(self) -> Path:
        return self.work_dir / "checkpoint.json"

    def _new_or_resume(self, resume: bool) -> dict[str, object]:
        if self.output_dir.exists() and not resume:
            raise FileExistsError(f"output directory already exists: {self.output_dir}")
        if self.work_dir.exists():
            if not resume or not self.checkpoint_path.exists():
                raise FileExistsError(
                    f"work directory exists; use resume=True with its checkpoint or remove it: {self.work_dir}")
            state = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
            if state.get("signature") != self._signature:
                raise ValueError("checkpoint inputs or counting configuration do not match")
            next_run = int(state["next_run_id"])
            for path in self.work_dir.glob("run_*"):
                try:
                    run_id = int(path.name.split("_")[1])
                except (ValueError, IndexError):
                    continue
                if run_id >= next_run:
                    path.unlink()
            for path in self.work_dir.glob("*.tmp"):
                path.unlink()
            return state
        self.work_dir.mkdir(parents=True)
        state: dict[str, object] = {
            "signature": self._signature,
            "next_offset": 0,
            "next_run_id": 0,
            "sentences": 0,
            "processed_ids": 0,
            "raw_occurrences": {str(order): 0 for order in range(1, self.max_order + 1)},
            "merge_completed_orders": [],
        }
        _atomic_json(self.checkpoint_path, state)
        return state

    def _flush(self, batches: dict[int, list[np.ndarray]], state: dict[str, object],
               next_offset: int, sentence_count: int) -> dict[str, object]:
        if not any(batches.values()):
            state["next_offset"] = next_offset
            state["sentences"] = sentence_count
            _atomic_json(self.checkpoint_path, state)
            return state
        run_id = int(state["next_run_id"])
        # Runs are committed first. A crash before checkpoint replacement is
        # recovered by deleting every run whose ID is not checkpointed.
        for order in range(1, self.max_order + 1):
            pieces = batches[order]
            if not pieces:
                continue
            raw = np.concatenate(pieces)
            keys, counts = np.unique(raw, return_counts=True)
            for name, values in (("keys", keys.astype("<u8", copy=False)),
                                 ("counts", counts.astype("<u8", copy=False))):
                path = self.work_dir / f"run_{run_id:08d}_order_{order}_{name}.npy"
                temporary = path.with_suffix(path.suffix + ".tmp")
                with temporary.open("wb") as file:
                    np.save(file, values, allow_pickle=False)
                os.replace(temporary, path)
            totals = state["raw_occurrences"]
            totals[str(order)] = int(totals[str(order)]) + len(raw)
        state["next_run_id"] = run_id + 1
        state["next_offset"] = next_offset
        state["sentences"] = sentence_count
        state["processed_ids"] = next_offset
        _atomic_json(self.checkpoint_path, state)
        for order in batches:
            batches[order].clear()
        return state

    def _count_runs(self, state: dict[str, object]) -> dict[str, object]:
        data = np.load(self.train_path, mmap_mode="r", allow_pickle=False)
        if data.ndim != 1 or data.dtype != np.int32:
            raise ValueError(f"expected flat int32 training array, found {data.shape} {data.dtype}")
        start_offset = int(state["next_offset"])
        scan_stop = int(data.size)
        if self.max_tokens is not None:
            inspected_stop = min(int(data.size), self.max_tokens)
            last_eos = -1
            for block_start in range(0, inspected_stop, self.scan_chunk_tokens):
                block_stop = min(block_start + self.scan_chunk_tokens, inspected_stop)
                found = np.flatnonzero(data[block_start:block_stop] == EOS)
                if found.size:
                    last_eos = block_start + int(found[-1])
            scan_stop = last_eos + 1 if last_eos >= 0 else 0
        if start_offset > scan_stop:
            raise ValueError("checkpoint offset exceeds requested input prefix")
        batches: dict[int, list[np.ndarray]] = {order: [] for order in range(1, self.max_order + 1)}
        batch_occurrences = 0
        active: list[int] | None = None
        active_start = 0
        completed_sentences = int(state["sentences"])
        processed = start_offset
        for chunk_start in range(start_offset, scan_stop, self.scan_chunk_tokens):
            chunk_stop = min(chunk_start + self.scan_chunk_tokens, scan_stop)
            chunk = data[chunk_start:chunk_stop]
            for raw_id in chunk:
                token_id = int(raw_id)
                offset = processed
                processed += 1
                if token_id == BOS:
                    if active is not None:
                        raise ValueError(f"nested <s> at input offset {offset}")
                    active = [BOS]
                    active_start = offset
                elif token_id == EOS:
                    if active is None:
                        raise ValueError(f"</s> outside sentence at input offset {offset}")
                    active.append(EOS)
                    if any(item < 0 or item >= len(self.vocabulary) for item in active):
                        raise ValueError("token ID outside vocabulary")
                    if any(item in {PAD, BOS, EOS} for item in active[1:-1]):
                        raise ValueError(f"structural token inside sentence ending at offset {offset}")
                    if self.max_sentence_tokens is not None and len(active) - 2 > self.max_sentence_tokens:
                        raise ValueError(f"sentence ending at offset {offset} exceeds configured token limit")
                    sentence_pieces = {order: _pack_sentence(active, order)
                                       for order in range(1, self.max_order + 1)}
                    sentence_occurrences = sum(map(len, sentence_pieces.values()))
                    if batch_occurrences and batch_occurrences + sentence_occurrences > self.run_occurrences_limit:
                        state = self._flush(batches, state, active_start, completed_sentences)
                        batch_occurrences = 0
                    for order, pieces in sentence_pieces.items():
                        if len(pieces):
                            batches[order].append(pieces)
                    batch_occurrences += sentence_occurrences
                    completed_sentences += 1
                    active = None
                    # Store a resumable boundary after a flush. Non-flushed
                    # complete sentences are deliberately replayed after a crash.
                    if batch_occurrences >= self.run_occurrences_limit:
                        state = self._flush(batches, state, processed, completed_sentences)
                        batch_occurrences = 0
                elif active is not None:
                    if token_id < 0 or token_id >= len(self.vocabulary):
                        raise ValueError(f"token ID outside vocabulary at input offset {offset}")
                    active.append(token_id)
                else:
                    raise ValueError(f"token outside sentence boundaries at input offset {offset}")
        # An unfinished tail from max_tokens is intentionally ignored: only
        # completed sentence sequences are eligible for counting.
        if active is not None:
            raise ValueError("training array ends with an unterminated sentence")
        state = self._flush(batches, state, processed, completed_sentences)
        state["processed_ids"] = int(state["next_offset"])
        state["scan_stop"] = scan_stop
        _atomic_json(self.checkpoint_path, state)
        return state

    def _merge_order(self, order: int, state: dict[str, object]) -> int:
        keys_out = self.output_dir / f"order_{order}_keys.u64"
        counts_out = self.output_dir / f"order_{order}_counts.u64"
        key_tmp = self.work_dir / f"order_{order}_keys.u64.tmp"
        count_tmp = self.work_dir / f"order_{order}_counts.u64.tmp"
        if order in state["merge_completed_orders"] and keys_out.exists() and counts_out.exists():
            return int(state["array_lengths"][str(order)]["keys"])
        key_runs, count_runs = [], []
        for run_id in range(int(state["next_run_id"])):
            key_path = self.work_dir / f"run_{run_id:08d}_order_{order}_keys.npy"
            count_path = self.work_dir / f"run_{run_id:08d}_order_{order}_counts.npy"
            if key_path.exists():
                key_runs.append(np.load(key_path, mmap_mode="r", allow_pickle=False))
                count_runs.append(np.load(count_path, mmap_mode="r", allow_pickle=False))
        heap: list[tuple[int, int, int]] = []
        positions = [0] * len(key_runs)
        for run_index, keys in enumerate(key_runs):
            if len(keys):
                heapq.heappush(heap, (int(keys[0]), run_index, int(count_runs[run_index][0])))
        key_buffer = np.empty(self.merge_buffer_size, dtype="<u8")
        count_buffer = np.empty(self.merge_buffer_size, dtype="<u8")
        used = 0
        merged = 0
        key_tmp.unlink(missing_ok=True)
        count_tmp.unlink(missing_ok=True)
        with key_tmp.open("wb") as kf, count_tmp.open("wb") as cf:
            while heap:
                key, run_index, count = heapq.heappop(heap)
                positions[run_index] += 1
                pos = positions[run_index]
                if pos < len(key_runs[run_index]):
                    heapq.heappush(heap, (int(key_runs[run_index][pos]), run_index,
                                          int(count_runs[run_index][pos])))
                while heap and heap[0][0] == key:
                    _, duplicate_run, duplicate_count = heapq.heappop(heap)
                    count += duplicate_count
                    positions[duplicate_run] += 1
                    duplicate_pos = positions[duplicate_run]
                    if duplicate_pos < len(key_runs[duplicate_run]):
                        heapq.heappush(heap, (int(key_runs[duplicate_run][duplicate_pos]),
                                              duplicate_run,
                                              int(count_runs[duplicate_run][duplicate_pos])))
                key_buffer[used] = key
                count_buffer[used] = count
                used += 1
                merged += 1
                if used == self.merge_buffer_size:
                    key_buffer.tofile(kf)
                    count_buffer.tofile(cf)
                    used = 0
            if used:
                key_buffer[:used].tofile(kf)
                count_buffer[:used].tofile(cf)
            kf.flush()
            cf.flush()
            os.fsync(kf.fileno())
            os.fsync(cf.fileno())
        os.replace(key_tmp, keys_out)
        os.replace(count_tmp, counts_out)
        return merged

    def _write_contexts(self, order: int, key_count: int) -> int:
        keys_path = self.output_dir / f"order_{order}_keys.u64"
        counts_path = self.output_dir / f"order_{order}_counts.u64"
        ctx_path = self.output_dir / f"order_{order}_context_keys.u64"
        sums_path = self.output_dir / f"order_{order}_context_totals.u64"
        if order == 1:
            ctx_path.touch()
            sums_path.touch()
            return 0
        keys = np.memmap(keys_path, dtype="<u8", mode="r", shape=(key_count,))
        counts = np.memmap(counts_path, dtype="<u8", mode="r", shape=(key_count,))
        # Contexts are prefixes and remain sorted. Keep one trailing group
        # across blocks so a prefix spanning a buffer edge is summed once.
        ctx_tmp, sums_tmp = self.work_dir / f"order_{order}_context_keys.tmp", self.work_dir / f"order_{order}_context_totals.tmp"
        pending_key: int | None = None
        pending_sum = 0
        buffer_ctx = np.empty(self.merge_buffer_size, dtype="<u8")
        buffer_sum = np.empty(self.merge_buffer_size, dtype="<u8")
        used = 0
        with ctx_tmp.open("wb") as cf, sums_tmp.open("wb") as sf:
            for start in range(0, key_count, self.merge_buffer_size):
                stop = min(start + self.merge_buffer_size, key_count)
                prefixes = keys[start:stop] >> np.uint64(16)
                starts = np.r_[0, np.flatnonzero(prefixes[1:] != prefixes[:-1]) + 1]
                sums = np.add.reduceat(counts[start:stop], starts, dtype=np.uint64)
                for prefix, value in zip(prefixes[starts], sums):
                    p = int(prefix)
                    if pending_key is not None and p != pending_key:
                        buffer_ctx[used], buffer_sum[used] = pending_key, pending_sum
                        used += 1
                        if used == self.merge_buffer_size:
                            buffer_ctx.tofile(cf)
                            buffer_sum.tofile(sf)
                            used = 0
                        pending_sum = 0
                    pending_key = p
                    pending_sum += int(value)
            if pending_key is not None:
                buffer_ctx[used], buffer_sum[used] = pending_key, pending_sum
                used += 1
            if used:
                buffer_ctx[:used].tofile(cf)
                buffer_sum[:used].tofile(sf)
        os.replace(ctx_tmp, ctx_path)
        os.replace(sums_tmp, sums_path)
        return int(ctx_path.stat().st_size // 8)

    def run(self, *, resume: bool = False, cleanup_runs: bool = False) -> DiskRunResult:
        """Count data and create a memory-mappable model artifact directory."""
        started = time.perf_counter()
        state = self._new_or_resume(resume)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        state = self._count_runs(state)
        state.setdefault("array_lengths", {})
        self.output_dir.joinpath("vocab.txt").write_text(
            "\n".join(self.vocabulary) + "\n", encoding="utf-8")
        unique_counts: dict[str, int] = {}
        for order in range(1, self.max_order + 1):
            merged = self._merge_order(order, state)
            contexts = self._write_contexts(order, merged)
            lengths = {"keys": merged, "counts": merged,
                       "context_keys": contexts, "context_totals": contexts}
            state["array_lengths"][str(order)] = lengths
            state["merge_completed_orders"] = sorted(set(state["merge_completed_orders"]) | {order})
            unique_counts[str(order)] = merged
            _atomic_json(self.checkpoint_path, state)
        metadata: dict[str, object] = {
            "format_version": 2,
            "storage_format": "little-endian raw uint64 arrays; memory-mapped",
            "max_order": self.max_order,
            "vocabulary_size": len(self.vocabulary),
            "preprocessing_rules_version": RULES_VERSION,
            "prediction_outcomes": "all vocabulary IDs except <PAD> and <s>; includes <UNK> and </s>",
            "counting": "sentence-local; <s> is context only; </s> is a prediction target",
            "array_lengths": state["array_lengths"],
            "source_train_path": str(self.train_path.resolve()),
            "source_ids_processed": int(state["next_offset"]),
            "complete_sentences": int(state["sentences"]),
            "raw_occurrences_by_order": state["raw_occurrences"],
            "unique_ngrams_by_order": unique_counts,
            "counting_configuration": self._signature,
        }
        manifest = self.output_dir / "metadata.json"
        _atomic_json(manifest, metadata)
        model = NGramModel.load(self.output_dir, mmap_mode="r")
        totals = {str(order): model.total_occurrences(order)
                  for order in range(1, self.max_order + 1)}
        expected = {str(order): int(state["raw_occurrences"][str(order)])
                    for order in range(1, self.max_order + 1)}
        if totals != expected:
            raise AssertionError(f"merged count totals differ from local runs: {totals} != {expected}")
        report = {
            "status": "ok",
            "source_ids_processed": int(state["next_offset"]),
            "complete_sentences": int(state["sentences"]),
            "raw_occurrences_by_order": totals,
            "unique_ngrams_by_order": unique_counts,
            "model_array_bytes": sum(path.stat().st_size for path in self.output_dir.glob("*.u64")),
            "work_run_bytes_before_cleanup": sum(path.stat().st_size for path in self.work_dir.iterdir()
                                                  if path.is_file()),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_local_occurrence_keys": self.run_occurrences_limit,
            "merge_buffer_entries": self.merge_buffer_size,
            "cleanup_runs_requested": cleanup_runs,
        }
        if cleanup_runs:
            shutil.rmtree(self.work_dir)
        return DiskRunResult(self.output_dir, manifest, report)
