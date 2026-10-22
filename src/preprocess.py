"""GPAC preprocessing pipeline (Task 1.1).

  python preprocess.py --src GPAC.txt --out processed_sample --sample 0.05
  python preprocess.py --src GPAC.txt --out processed_full   --sample 1.0

Output folder: train.npy dev.npy test.npy (int32 ID arrays), vocab.txt (line number = ID),
stats.json (rules, seeds and numbers) and longest_50.txt (for manual inspection)."""
import argparse
import heapq
import json
import os
import re
import sys
import time
from array import array
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess_lib import (MERGE_FOURTH_ORDER_HA, RULES_VERSION, SAMPLE_SEED, SPLIT_RE, SPLIT_SEED,
                            hash64, sentence_key, split_of_hash, tokenize)

CHUNK_CHARS = 16_000_000
MAX_TOKENS = 100                # sentences longer than this (after tokenization) are dropped
VOCAB_SIZE = 50_000             # including the 4 special tokens
LONGEST_KEEP = 50
DEDUP_DEV_TEST = False          # True: drop repeated sentences inside dev and test
SPLITS = ("train", "dev", "test")
NORMALIZATION_EXAMPLES = ["ሲጠጣ፣ በሩቅ እያየች", "የአለም የእግር ኳስ ዋ ለ19ኛ ጊዜ", "በ1980 ዓ.ም",
                          "መጽሐፍ ሐምሌ ዓለም ሠላም", "ሃገር ሓገር ኃይል ሀገር"]
SPECIALS = ["<PAD>", "<UNK>", "<s>", "</s>"]
PAD, UNK, BOS, EOS = 0, 1, 2, 3


def iter_pieces(path, chunk_chars=CHUNK_CHARS):
    """Yield raw sentence pieces without ever loading the whole file.
    The unfinished text at the end of a chunk is carried into the next chunk,
    so no sentence (and no delimiter such as ፡፡) is cut at a chunk boundary."""
    carry = ""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        while True:
            chunk = f.read(chunk_chars)
            if not chunk:
                break
            pieces = SPLIT_RE.split(carry + chunk)
            carry = pieces.pop()
            yield from pieces
    if carry:
        yield carry


def pass1(src, chunk, sample, tmp_paths):
    """Read the corpus once: sample sentences, tokenize, filter, assign splits,
    write each split's tokens to a temporary text file and count train words."""
    t0 = time.time()
    c = Counter()
    train_counts = Counter()
    other_types = set()
    seen = set()                       # hashes of sentences already seen (duplicate statistics)
    longest = []                       # min-heap of the longest dropped segments
    files = {s: open(tmp_paths[s], "w", encoding="utf-8") for s in SPLITS}
    try:
        for piece in iter_pieces(src, chunk):
            piece = piece.strip()
            if not piece:
                continue
            c["pieces"] += 1
            c["raw_words"] += len(piece.split())
            if c["pieces"] % 500_000 == 0:
                print(f"  pass 1: {c['pieces']:,} pieces read ({time.time() - t0:.0f}s)", flush=True)

            # whole-sentence sampling with a fixed seed
            if sample < 1.0 and hash64(SAMPLE_SEED, piece) / 2**64 >= sample:
                continue
            c["pieces_sampled"] += 1

            toks = tokenize(piece)
            if not toks:
                c["empty_after_tokenize"] += 1
                continue
            if len(toks) > MAX_TOKENS:
                c["dropped_long"] += 1
                c["tokens_discarded"] += len(toks)
                item = (len(toks), len(re.findall(r" {5,}", piece)), piece[:600])
                if len(longest) < LONGEST_KEEP:
                    heapq.heappush(longest, item)
                else:
                    heapq.heappushpop(longest, item)
                continue

            key = sentence_key(toks)
            h = hash64(SPLIT_SEED, key)
            split = split_of_hash(h)
            if h in seen:
                c[f"duplicates_{split}"] += 1
                if DEDUP_DEV_TEST and split != "train":
                    c["removed_duplicates"] += 1
                    continue
            else:
                seen.add(h)

            files[split].write(key + "\n")
            c[f"sentences_{split}"] += 1
            if split == "train":
                train_counts.update(toks)
            else:
                other_types.update(toks)
    finally:
        for f in files.values():
            f.close()
    return c, train_counts, other_types, longest


def build_vocab(train_counts):
    """Vocabulary from the TRAIN split only: 4 special tokens + the most frequent words."""
    ordered = sorted(train_counts.items(), key=lambda kv: (-kv[1], kv[0]))   # deterministic ties
    return SPECIALS + [w for w, _ in ordered[: VOCAB_SIZE - len(SPECIALS)]]


def pass2(vocab, tmp_paths):
    """Turn each split's token file into one int32 array: <s> w1 ... wn </s> <s> ..."""
    assert array("i").itemsize == 4, "int32 expected"
    get = {w: i for i, w in enumerate(vocab)}.get
    data = {}
    for split in SPLITS:
        arr = array("i")
        with open(tmp_paths[split], "r", encoding="utf-8") as f:
            for line in f:
                key = line.rstrip("\n")
                assert split_of_hash(hash64(SPLIT_SEED, key)) == split, "sentence in the wrong split"
                arr.append(BOS)
                arr.extend([get(t, UNK) for t in key.split()])
                arr.append(EOS)
        data[split] = np.frombuffer(arr, dtype=np.int32) if len(arr) else np.zeros(0, dtype=np.int32)
    return data


def validate_output(out_dir):
    """Checks that must hold for every run. Raises AssertionError if one fails.
    An empty split is reported clearly; its other checks are skipped only where they cannot apply."""
    out_dir = Path(out_dir)
    vocab = (out_dir / "vocab.txt").read_text(encoding="utf-8").split("\n")
    stats = json.loads((out_dir / "stats.json").read_text(encoding="utf-8"))
    assert vocab[:4] == SPECIALS, "special tokens must be IDs 0-3"
    assert len(set(vocab)) == len(vocab), "duplicate words in vocabulary"
    assert len(vocab) <= VOCAB_SIZE
    arrays = {s: np.load(out_dir / f"{s}.npy") for s in SPLITS}
    for split, arr in arrays.items():
        assert arr.dtype == np.int32, f"{split}: dtype must be int32"
        if arr.size == 0:
            print(f"WARNING: the {split} split is empty (0 sentences).")
            assert stats["sentences"][split] == 0, f"{split} array is empty but stats.json says otherwise"
            continue
        assert arr.min() >= 0 and arr.max() < len(vocab), "ID outside the vocabulary"
        assert arr[0] == BOS and arr[-1] == EOS
        marks = arr[(arr == BOS) | (arr == EOS)]
        assert len(marks) % 2 == 0 and (marks[0::2] == BOS).all() and (marks[1::2] == EOS).all(), \
            "<s> and </s> are not paired"
        assert int((arr == BOS).sum()) == stats["sentences"][split]
        assert int((arr == PAD).sum()) == 0
    assert arrays["train"].size > 0, "the train split is empty, so no vocabulary can be built"
    train_counts = np.bincount(arrays["train"], minlength=len(vocab))
    assert (train_counts[4:] > 0).all(), "vocabulary contains a word that is not in train"
    return vocab, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=float, required=True, help="fraction of sentences to use: 0.05 or 1.0")
    ap.add_argument("--chunk", type=int, default=CHUNK_CHARS)
    args = ap.parse_args()
    assert 0 < args.sample <= 1.0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp_paths = {s: out / f"tokens_{s}.tmp" for s in SPLITS}
    t0 = time.time()
    print(f"Source: {args.src}\nSample fraction: {args.sample}\nOutput: {out}", flush=True)

    c, train_counts, other_types, longest = pass1(args.src, args.chunk, args.sample, tmp_paths)
    print(f"Pass 1 done ({time.time() - t0:.0f}s): {c['pieces']:,} pieces, "
          f"{c['pieces_sampled']:,} sampled", flush=True)

    vocab = build_vocab(train_counts)
    data = pass2(vocab, tmp_paths)
    print(f"Pass 2 done ({time.time() - t0:.0f}s). Vocabulary size: {len(vocab):,}", flush=True)

    n_sent = {s: int((data[s] == BOS).sum()) for s in SPLITS}
    n_tok = {s: int(len(data[s]) - 2 * n_sent[s]) for s in SPLITS}
    n_unk = {s: int((data[s] == UNK).sum()) for s in SPLITS}
    for s in SPLITS:
        assert n_sent[s] == c[f"sentences_{s}"], "sentence count mismatch"
        np.save(out / f"{s}.npy", data[s])
    (out / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    for p in tmp_paths.values():
        p.unlink()

    with open(out / "longest_50.txt", "w", encoding="utf-8") as f:
        for n, runs, text in sorted(longest, reverse=True):
            f.write(f"[{n} tokens | {runs} runs of 5+ spaces]\n{text}\n\n")

    spelling = ("NFC; ሐ ኀ -> ሀ; ሠ -> ሰ; ዐ ዓ ኣ -> አ; ፀ -> ጸ "
                "(all 7 vowel orders; the 4th order of ዐ and አ also -> አ)")
    if MERGE_FOURTH_ORDER_HA:
        spelling += "; ሃ ሓ ኃ -> ሀ"
    kept = sum(n_tok.values())
    discarded = c["tokens_discarded"]
    report = {
        "version": RULES_VERSION,
        "source": args.src,
        "sample_fraction": args.sample,
        "seeds": {"sample": SAMPLE_SEED, "split": SPLIT_SEED},
        "rules": {
            "sentence_delimiters": ["፡፡", "::", "።", "?", "!", "፧"],
            "spelling": spelling,
            "punctuation": "dropped; only <s> and </s> mark sentence boundaries",
            "numbers": "digits or Ethiopic numerals -> <NUM>, attached letters kept (19ኛ -> <NUM>ኛ, "
                       "በ1980 -> በ<NUM>); identical consecutive numbers collapse; different numbers do not",
            "max_tokens_per_sentence": MAX_TOKENS,
            "split": "80/10/10 by hash of the normalized sentence (blake2b, seed in 'seeds')",
            "vocabulary": f"top {VOCAB_SIZE:,} incl. 4 special tokens, from train only",
            "dedup_dev_test": DEDUP_DEV_TEST,
        },
        "corpus": {"pieces_total": c["pieces"], "raw_words_total": c["raw_words"],
                   "pieces_sampled": c["pieces_sampled"]},
        "sentences": n_sent,
        "tokens": n_tok,
        "unk_tokens": n_unk,
        "unk_rate_percent": {s: round(100 * n_unk[s] / n_tok[s], 2) if n_tok[s] else 0.0 for s in SPLITS},
        "dropped_too_long": c["dropped_long"],
        "tokens_discarded": discarded,
        "tokens_discarded_percent": round(100 * discarded / (kept + discarded), 2) if kept + discarded else 0.0,
        "empty_after_tokenize": c["empty_after_tokenize"],
        "duplicate_sentences": {s: c[f"duplicates_{s}"] for s in SPLITS},
        "removed_duplicates": c["removed_duplicates"],
        "vocab": {"size": len(vocab), "unique_words_train": len(train_counts),
                  "unique_words_total": len(set(train_counts) | other_types)},
        "special_ids": {"<PAD>": PAD, "<UNK>": UNK, "<s>": BOS, "</s>": EOS},
        "normalization_examples": [{"raw": s, "tokens": tokenize(s)} for s in NORMALIZATION_EXAMPLES],
        "seconds": round(time.time() - t0),
    }
    (out / "stats.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    validate_output(out)
    print("\nAll output checks passed.")


if __name__ == "__main__":
    main()
