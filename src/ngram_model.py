"""Sparse, count-based N-gram language models for orders one through four.

N-grams are stored as sorted packed ``uint64`` keys rather than dense
vocabulary-sized matrices or Python tuple dictionaries. Sentence inputs remain
separate while counting, and each sentence gets virtual BOS context padding,
so no N-gram can cross an EOS boundary.

The current in-memory ``fit`` implementation is intended for toy and bounded
sample runs. Full-corpus fitting needs an external/disk-backed merge strategy
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from src.preprocess_lib import RULES_VERSION, tokenize as preprocess_tokenize

PAD, UNK, BOS, EOS = 0, 1, 2, 3
SPECIAL_TOKENS = {"<PAD>", "<UNK>", "<s>", "</s>"}
NON_PREDICTABLE_IDS = {PAD, BOS}


def normalize_and_tokenize(text: str) -> list[str]:
    """Use the same normalization and tokenizer as preprocessing."""
    return preprocess_tokenize(text)


def iter_encoded_sentences(flat_ids: Sequence[int] | np.ndarray) -> Iterable[list[int]]:
    """Yield BOS...EOS sentences and reject malformed flattened streams."""
    active = False
    sentence: list[int] = []
    for raw_id in flat_ids:
        token_id = int(raw_id)
        if token_id == BOS:
            if active:
                raise ValueError("nested <s> before </s>")
            active = True
            sentence = [BOS]
        elif token_id == EOS:
            if not active:
                raise ValueError("</s> without a preceding <s>")
            sentence.append(EOS)
            yield sentence
            sentence = []
            active = False
        elif token_id == PAD:
            raise ValueError("<PAD> is not valid in the unpadded sentence stream")
        elif active:
            sentence.append(token_id)
        else:
            raise ValueError("token outside sentence boundaries")
    if active:
        raise ValueError("unterminated sentence: missing </s>")


def _pack_windows(tokens: np.ndarray, order: int) -> np.ndarray:
    """Pack rows of up to four 16-bit token IDs into uint64 keys."""
    windows = np.lib.stride_tricks.sliding_window_view(tokens, order)
    packed = np.zeros(len(windows), dtype=np.uint64)
    for col in range(order):
        packed |= windows[:, col].astype(np.uint64) << np.uint64(16 * (order - col - 1))
    return packed


def _pack_ids(token_ids: Sequence[int]) -> np.uint64:
    key = np.uint64(0)
    for token_id in token_ids:
        key = (key << np.uint64(16)) | np.uint64(token_id)
    return key


@dataclass
class OrderCounts:
    """Sorted sparse counts for one N-gram order and its contexts."""

    order: int
    keys: np.ndarray
    counts: np.ndarray
    context_keys: np.ndarray
    context_totals: np.ndarray

    def count(self, token_ids: Sequence[int]) -> int:
        if len(token_ids) != self.order:
            return 0
        key = _pack_ids(token_ids)
        index = int(np.searchsorted(self.keys, key))
        return int(self.counts[index]) if index < len(self.keys) and self.keys[index] == key else 0

    def context_count(self, context: Sequence[int]) -> int:
        if self.order == 1 or len(context) != self.order - 1:
            return 0
        key = _pack_ids(context)
        index = int(np.searchsorted(self.context_keys, key))
        return (int(self.context_totals[index])
                if index < len(self.context_keys) and self.context_keys[index] == key else 0)

    def next_counts(self, context: Sequence[int]) -> list[tuple[int, int]]:
        """Return observed ``(next token ID, count)`` pairs for one context."""
        if self.order == 1 or len(context) != self.order - 1:
            return []
        prefix = _pack_ids(context)
        shift = np.uint64(16)
        lo = int(np.searchsorted(self.keys, prefix << shift))
        hi = int(np.searchsorted(self.keys, (prefix + np.uint64(1)) << shift))
        keys = self.keys[lo:hi]
        counts = self.counts[lo:hi]
        mask = (keys >> shift) == prefix
        return [(int(key & np.uint64(0xFFFF)), int(count))
                for key, count in zip(keys[mask], counts[mask])]


class NGramModel:
    """Unsmoothed, add-k, linearly interpolated, and backed-off N-gram model."""

    def __init__(self, vocabulary: Sequence[str], orders: dict[int, OrderCounts]):
        self.vocabulary = list(vocabulary)
        self.word_to_id = {word: i for i, word in enumerate(self.vocabulary)}
        self.orders = orders
        self.max_order = max(orders, default=1)
        # Prediction events include ordinary words, UNK, and EOS, but not PAD/BOS.
        self.allowed_outcomes = [i for i in range(len(self.vocabulary))
                                 if i not in NON_PREDICTABLE_IDS]
        self.allowed_outcome_set = set(self.allowed_outcomes)
        self._backoff_alpha_cache: dict[tuple[int, ...], float | None] = {}

    @classmethod
    def fit(cls, sentences: Iterable[Sequence[int]], vocabulary: Sequence[str],
            max_order: int = 4, max_occurrences: int = 10_000_000) -> "NGramModel":
        """Count sentence-framed training data into sparse packed-key tables.

        The input is validated sentence-by-sentence and stored as packed sparse
        keys. ``max_occurrences`` caps the total raw keys retained across all
        orders; use a disk-backed merge for full-corpus fitting rather than
        raising this guard blindly.
        """
        if len(vocabulary) < 4 or list(vocabulary[:4]) != ["<PAD>", "<UNK>", "<s>", "</s>"]:
            raise ValueError("vocabulary must start with <PAD>, <UNK>, <s>, </s> at IDs 0-3")
        if len(vocabulary) >= 65536:
            raise ValueError("packed keys require vocabulary size below 65,536")
        if len(set(vocabulary)) != len(vocabulary):
            raise ValueError("vocabulary entries must be unique")
        if max_order < 1 or max_order > 4:
            raise ValueError("max_order must be between 1 and 4")
        if max_occurrences < 1:
            raise ValueError("max_occurrences must be a positive integer")

        # A list per order is compact NumPy data, unlike millions of Python
        # tuple keys. The hard cap bounds these temporary arrays; full-corpus
        # work needs an external merge so the cap need not be raised.
        occurrences: dict[int, list[np.ndarray]] = {order: [] for order in range(1, max_order + 1)}
        occurrence_total = 0
        for sentence in sentences:
            raw = np.asarray(sentence)
            if raw.ndim != 1 or not len(raw):
                raise ValueError("each training sentence must be a non-empty 1-D ID sequence")
            if not np.issubdtype(raw.dtype, np.integer):
                raise ValueError("training token IDs must be integers")
            if int(raw.min()) < 0 or int(raw.max()) >= len(vocabulary):
                raise ValueError("training token ID is outside the vocabulary")
            framed = raw.astype(np.uint16, copy=False)
            if len(framed) < 2 or int(framed[0]) != BOS or int(framed[-1]) != EOS:
                raise ValueError("each training sentence must include <s> and </s>")
            if any(int(token) in {PAD, BOS, EOS} for token in framed[1:-1]):
                raise ValueError("sentence contents cannot contain structural token IDs")
            predicted_events = len(framed) - 1  # each word plus one EOS target
            if occurrence_total + predicted_events * max_order > max_occurrences:
                raise MemoryError(
                    f"fit would exceed max_occurrences={max_occurrences:,}; "
                    "use a smaller complete-sentence slice or disk-backed counting"
                )

            # Do not join sentences. For each order, prepend n-1 virtual BOS
            # markers so every word and EOS has one prediction event.
            for order in range(1, max_order + 1):
                if order == 1:
                    tokens = framed[1:]  # ordinary words plus EOS; BOS is context only
                    keys = tokens.astype(np.uint64)
                else:
                    tokens = np.concatenate((np.full(order - 1, BOS, dtype=np.uint16), framed[1:]))
                    keys = _pack_windows(tokens, order) if len(tokens) >= order else np.empty(0, np.uint64)
                if len(keys):
                    occurrence_total += len(keys)
                    if occurrence_total > max_occurrences:
                        raise MemoryError(
                            f"fit exceeded max_occurrences={max_occurrences:,}; "
                            "use a smaller complete-sentence slice or disk-backed counting"
                        )
                    occurrences[order].append(keys)

        orders: dict[int, OrderCounts] = {}
        for order, chunks in occurrences.items():
            all_keys = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.uint64)
            keys, counts = np.unique(all_keys, return_counts=True)
            if order == 1:
                context_keys = np.empty(0, dtype=np.uint64)
                context_totals = np.empty(0, dtype=np.uint64)
            else:
                context_keys, first = np.unique(keys >> np.uint64(16), return_index=True)
                context_totals = np.add.reduceat(counts.astype(np.uint64), first)
            orders[order] = OrderCounts(order, keys, counts.astype(np.uint64),
                                        context_keys, context_totals)
        return cls(vocabulary, orders)

    @classmethod
    def from_flat_ids(cls, flat_ids: Sequence[int] | np.ndarray, vocabulary: Sequence[str],
                      max_order: int = 4, max_occurrences: int = 10_000_000) -> "NGramModel":
        """Decode a flattened BOS/EOS stream and fit without crossing boundaries."""
        return cls.fit(iter_encoded_sentences(flat_ids), vocabulary, max_order, max_occurrences)

    def save(self, output_dir: str | Path) -> Path:
        """Save sparse model arrays and metadata into a new directory."""
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=False)
        (output / "vocab.txt").write_text("\n".join(self.vocabulary) + "\n", encoding="utf-8")
        for order, counts in self.orders.items():
            np.save(output / f"order_{order}_keys.npy", counts.keys, allow_pickle=False)
            np.save(output / f"order_{order}_counts.npy", counts.counts, allow_pickle=False)
            np.save(output / f"order_{order}_context_keys.npy", counts.context_keys, allow_pickle=False)
            np.save(output / f"order_{order}_context_totals.npy", counts.context_totals, allow_pickle=False)
        metadata = {
            "format_version": 1,
            "max_order": self.max_order,
            "vocabulary_size": len(self.vocabulary),
            "preprocessing_rules_version": RULES_VERSION,
            "prediction_outcomes": "all vocabulary IDs except <PAD> and <s>; includes <UNK> and </s>",
            "counting": "sentence-local; <s> is context only; </s> is a prediction target",
        }
        (output / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return output

    @classmethod
    def load(cls, model_dir: str | Path, mmap_mode: str | None = "r") -> "NGramModel":
        """Load saved sparse arrays, memory-mapping them by default."""
        source = Path(model_dir)
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        format_version = metadata.get("format_version")
        if format_version not in {1, 2}:
            raise ValueError("unsupported model artifact format")
        vocabulary = (source / "vocab.txt").read_text(encoding="utf-8").splitlines()
        if len(vocabulary) != metadata.get("vocabulary_size"):
            raise ValueError("model vocabulary size does not match metadata")
        if vocabulary[:4] != ["<PAD>", "<UNK>", "<s>", "</s>"]:
            raise ValueError("saved model has invalid special-token IDs")
        max_order = int(metadata["max_order"])
        if max_order < 1 or max_order > 4:
            raise ValueError("saved model has invalid max_order")
        orders = {}
        for order in range(1, max_order + 1):
            names = ("keys", "counts", "context_keys", "context_totals")
            if format_version == 1:
                arrays = [np.load(source / f"order_{order}_{name}.npy", mmap_mode=mmap_mode,
                                  allow_pickle=False) for name in names]
            else:
                lengths = metadata["array_lengths"][str(order)]
                arrays = []
                for name in names:
                    length = int(lengths[name])
                    if length == 0:
                        arrays.append(np.empty(0, dtype=np.uint64))
                    else:
                        arrays.append(np.memmap(source / f"order_{order}_{name}.u64",
                                                dtype="<u8", mode=mmap_mode or "r",
                                                shape=(length,)))
            orders[order] = OrderCounts(order, *arrays)
        return cls(vocabulary, orders)

    def ngram_count(self, *tokens: str) -> int:
        ids = [self.word_to_id.get(token, UNK) for token in tokens]
        if not ids or len(ids) not in self.orders:
            return 0
        return self.orders[len(ids)].count(ids)

    def ngram_count_ids(self, token_ids: Sequence[int]) -> int:
        if not token_ids or len(token_ids) not in self.orders:
            return 0
        return self.orders[len(token_ids)].count(token_ids)

    def total_occurrences(self, order: int) -> int:
        return int(self.orders[order].counts.sum(dtype=np.uint64))

    def unique_ngrams(self, order: int) -> int:
        return len(self.orders[order].keys)

    def top_ngrams(self, order: int, k: int = 10) -> list[tuple[tuple[str, ...], int]]:
        counts = self.orders[order].counts
        indexes = np.argsort(counts)[::-1][:max(0, k)]
        result = []
        for index in indexes:
            key = int(self.orders[order].keys[index])
            ids = [(key >> (16 * shift)) & 0xFFFF for shift in range(order - 1, -1, -1)]
            result.append((tuple(self.vocabulary[token_id] for token_id in ids), int(counts[index])))
        return result

    def _validate_event(self, word_id: int, context_ids: Sequence[int]) -> tuple[int, list[int]]:
        word_id = int(word_id)
        context = [int(token_id) for token_id in context_ids]
        if word_id < 0 or word_id >= len(self.vocabulary):
            raise ValueError("word ID is outside the vocabulary")
        if any(token_id < 0 or token_id >= len(self.vocabulary) for token_id in context):
            raise ValueError("context ID is outside the vocabulary")
        context = context[-(self.max_order - 1):] if self.max_order > 1 else []
        return word_id, context

    def _mle_probability(self, word_id: int, context: Sequence[int]) -> float:
        if word_id not in self.allowed_outcome_set:
            return 0.0
        order = len(context) + 1
        counts = self.orders[order]
        numerator = counts.count([*context, word_id])
        denominator = (self.total_occurrences(1) if order == 1
                       else counts.context_count(context))
        return numerator / denominator if denominator else 0.0

    def _add_k_probability(self, word_id: int, context: Sequence[int], k: float) -> float:
        if word_id not in self.allowed_outcome_set:
            return 0.0
        order = len(context) + 1
        counts = self.orders[order]
        numerator = counts.count([*context, word_id])
        denominator = (self.total_occurrences(1) if order == 1
                       else counts.context_count(context))
        smoothed_denominator = denominator + k * len(self.allowed_outcomes)
        # k=0 is explicitly supported as the unsmoothed limit. An unseen
        # context then has a zero denominator, so every continuation has
        # probability zero, matching the MLE behavior.
        return ((numerator + k) / smoothed_denominator
                if smoothed_denominator else 0.0)

    def _backoff_probability(self, word_id: int, context: Sequence[int]) -> float:
        """Absolute-discount backoff with a normalized lower-order escape mass.

        Seen N-grams receive ``max(count - 0.5, 0) / context_count``. The
        removed probability mass is allocated to lower-order probabilities
        over outcomes unseen after this context. An unseen context backs off
        directly to its suffix context.
        """
        if word_id not in self.allowed_outcome_set:
            return 0.0
        if not context:
            return self._mle_probability(word_id, [])

        context = list(context)
        order_counts = self.orders[len(context) + 1]
        denominator = order_counts.context_count(context)
        suffix = context[1:]
        if denominator == 0:
            return self._backoff_probability(word_id, suffix)

        count = order_counts.count([*context, word_id])
        discount = 0.5
        context_key = tuple(context)
        if self._backoff_alpha_cache.get(context_key, "missing") is None:
            observed = order_counts.next_counts(context)
            discounted_mass = sum(max(n - discount, 0.0) / denominator for _, n in observed)
            return ((max(count - discount, 0.0) / denominator) / discounted_mass
                    if count and discounted_mass else 0.0)
        if count:
            return max(count - discount, 0.0) / denominator

        if context_key not in self._backoff_alpha_cache:
            observed = order_counts.next_counts(context)
            discounted_mass = sum(max(n - discount, 0.0) / denominator
                                  for _, n in observed)
            lower_seen_mass = sum(self._backoff_probability(token_id, suffix)
                                  for token_id, _ in observed)
            lower_unseen_mass = max(0.0, 1.0 - lower_seen_mass)
            if lower_unseen_mass <= 1e-15:
                # The lower-order model gives no mass to any unseen outcome;
                # in that degenerate case, retain the normalized MLE row.
                alpha = None
            else:
                alpha = max(0.0, 1.0 - discounted_mass) / lower_unseen_mass
            if len(self._backoff_alpha_cache) >= 4096:
                self._backoff_alpha_cache.clear()
            self._backoff_alpha_cache[context_key] = alpha
        alpha = self._backoff_alpha_cache[context_key]
        if alpha is None:
            return 0.0
        return alpha * self._backoff_probability(word_id, suffix)

    def _interpolated_probability(self, word_id: int, context: Sequence[int],
                                  weights: Sequence[float] | None) -> float:
        levels = len(context) + 1
        if weights is None:
            weights = [1.0 / levels] * levels
        if len(weights) != levels or any(weight < 0 for weight in weights):
            raise ValueError("interpolation weights must be non-negative and match available orders")
        total_weight = sum(weights)
        if not np.isclose(total_weight, 1.0):
            raise ValueError("interpolation weights must sum to 1")
        # Use the MLE for every seen component context. If a component history
        # is absent, back it off so the mixture remains a normalized
        # distribution for an unseen full history.
        result = 0.0
        for order, weight in enumerate(weights, start=1):
            suffix = context[-(order - 1):] if order > 1 else []
            if order == 1 or self.orders[order].context_count(suffix):
                component = self._mle_probability(word_id, suffix)
            else:
                component = self._backoff_probability(word_id, suffix)
            result += weight * component
        return result

    def prob(self, word_id: int, context_ids: Sequence[int] = (), *,
             method: str = "mle", k: float = 1.0,
             weights: Sequence[float] | None = None) -> float:
        """Return ``P(word_id | context_ids)``.

        ``method`` is ``mle``, ``add_k``, ``interpolation``, or ``backoff``.
        Prediction outcomes exclude PAD and BOS but include UNK and EOS. Add-k
        uses ``k`` pseudo-counts for each such outcome, so its denominator is
        context count + k * outcome-vocabulary size.
        """
        word_id, context = self._validate_event(word_id, context_ids)
        if method == "mle":
            return self._mle_probability(word_id, context)
        if method == "add_k":
            if k < 0:
                raise ValueError("add-k smoothing requires k >= 0")
            return self._add_k_probability(word_id, context, k)
        if method == "backoff":
            return self._backoff_probability(word_id, context)
        if method == "interpolation":
            return self._interpolated_probability(word_id, context, weights)
        raise ValueError("method must be one of: mle, add_k, interpolation, backoff")

    def logprob(self, word_id: int, context_ids: Sequence[int] = (), **kwargs) -> float:
        """Return the natural logarithm of ``prob``; zero probability maps to -inf."""
        probability = self.prob(word_id, context_ids, **kwargs)
        return float(np.log(probability)) if probability > 0 else float("-inf")

    def conditional_probability(self, context: Sequence[str], next_word: str,
                                smoothing: float = 0.0) -> float:
        """String-token convenience wrapper retained for existing notebook code."""
        if smoothing < 0:
            raise ValueError("smoothing must be non-negative")
        context_ids = [self.word_to_id.get(word, UNK) for word in context]
        word_id = self.word_to_id.get(next_word, UNK)
        method = "mle" if smoothing == 0 else "add_k"
        return self.prob(word_id, context_ids, method=method, k=smoothing)

    def conditional_distribution(self, context: Sequence[str], smoothing: float = 0.0,
                                 method: str | None = None) -> dict[str, float]:
        """Return the full outcome distribution, including EOS and UNK."""
        context_ids = [self.word_to_id.get(word, UNK) for word in context]
        chosen = method or ("mle" if smoothing == 0 else "add_k")
        return {self.vocabulary[token_id]: self.prob(token_id, context_ids,
                                                      method=chosen, k=smoothing)
                for token_id in self.allowed_outcomes
                if self.prob(token_id, context_ids, method=chosen, k=smoothing) > 0}

    def sentence_probability(self, tokens: Sequence[str], order: int = 2,
                             smoothing: float = 0.0, method: str | None = None) -> float:
        if order not in self.orders:
            raise ValueError("requested order is not available in this model")
        ids = [self.word_to_id.get(token, UNK) for token in tokens]
        framed = [BOS] * (order - 1) + ids + [EOS]
        chosen = method or ("mle" if smoothing == 0 else "add_k")
        probability = 1.0
        for i in range(order - 1, len(framed)):
            context = framed[i - order + 1:i]
            probability *= self.prob(framed[i], context, method=chosen, k=smoothing)
        return probability

    def predict_next(self, context_words: str | Sequence[str], k: int = 5,
                     method: str = "backoff") -> list[tuple[str, float]]:
        """Rank ordinary-word suggestions by conditional probability.

        Scores are probabilities from the selected model distribution, not
        renormalized over only the displayed top-k suggestions.
        """
        if k <= 0:
            return []
        if isinstance(context_words, str):
            words = normalize_and_tokenize(context_words)
        else:
            words = [token for word in context_words for token in normalize_and_tokenize(word)]
        context_ids = [self.word_to_id.get(word, UNK) for word in words]
        scored = []
        for token_id in self.allowed_outcomes:
            token = self.vocabulary[token_id]
            if token in SPECIAL_TOKENS:
                continue
            score = self.prob(token_id, context_ids, method=method)
            if score > 0:
                scored.append((token, score))
        return sorted(scored, key=lambda item: (-item[1], item[0]))[:k]

    def storage_bytes(self) -> int:
        return sum(array.nbytes for order in self.orders.values()
                   for array in (order.keys, order.counts, order.context_keys, order.context_totals))


def encode_text_tokens(tokens: Sequence[str], word_to_id: dict[str, int]) -> list[int]:
    return [BOS, *(word_to_id.get(token, UNK) for token in tokens), EOS]


def count_token_frequencies(sentences: Iterable[Sequence[int]]) -> Counter[int]:
    """Count token IDs, excluding structural BOS/PAD and retaining EOS/UNK."""
    counts: Counter[int] = Counter()
    for sentence in sentences:
        counts.update(token_id for token_id in sentence if token_id not in {PAD, BOS})
    return counts
