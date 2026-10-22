"""Shared text rules for GPAC preprocessing, predict_next() and the app.
Everything must import these functions. Never re-implement them elsewhere."""
import hashlib
import re
import unicodedata

RULES_VERSION = "preprocessing-v2"

# ------------------------------------------------------------------ settings
MERGE_FOURTH_ORDER_HA = True   # True also merges ሃ ሓ ኃ -> ሀ (like ዓ ኣ -> አ). Team decision.
NUM = "<NUM>"
SAMPLE_SEED = 42                # picks which sentences go into the sample
SPLIT_SEED = 7                  # decides train / dev / test for every sentence

# Sentence ends: ፡፡  ::  ።  ?  !  ፧   (never '.')
SPLIT_RE = re.compile(r"(?:፡፡|::|[።?!፧])+")

# ------------------------------------------------------------------ spelling variants
_ORDERS = ["A", "U", "I", "AA", "EE", "E", "O"]   # the 7 vowel orders of an Ethiopic letter


def _syl(prefix, order):
    """Look a syllable up by its Unicode name, so a wrong name fails loudly."""
    return unicodedata.lookup(f"ETHIOPIC SYLLABLE {prefix}{order}")


def _build_variant_table():
    table = {}
    # variant family -> standard family, all 7 orders of each
    for src, dst in [("HH", "H"), ("X", "H"),        # ሐ ኀ -> ሀ
                     ("SZ", "S"),                    # ሠ   -> ሰ
                     ("PHARYNGEAL ", "GLOTTAL "),    # ዐ   -> አ
                     ("TZ", "TS")]:                  # ፀ   -> ጸ
        for o in _ORDERS:
            table[ord(_syl(src, o))] = _syl(dst, o)
    # 4th order merged into the 1st order as well: ዓ ኣ -> አ
    a = _syl("GLOTTAL ", "A")
    table[ord(_syl("PHARYNGEAL ", "AA"))] = a
    table[ord(_syl("GLOTTAL ", "AA"))] = a
    if MERGE_FOURTH_ORDER_HA:
        ha = _syl("H", "A")
        for fam in ("H", "HH", "X"):
            table[ord(_syl(fam, "AA"))] = ha         # ሃ ሓ ኃ -> ሀ
    return table


_VARIANTS = _build_variant_table()


def normalize(text):
    """Spelling normalization only: Unicode NFC + merging of letter variants."""
    return unicodedata.normalize("NFC", text).translate(_VARIANTS)


# ------------------------------------------------------------------ tokenizer
_LETTER = r"[^\W\d_\u1369-\u137C]"                    # letters (not digits, not Ethiopic numerals)
_DIGITS = r"[0-9\u1369-\u137C]"                       # Arabic digits + Ethiopic numerals
_NUMBER = rf"<NUM>|{_DIGITS}+(?:[.,]{_DIGITS}+)*"     # accepts <NUM> itself, so tokenize() is idempotent
TOKEN_RE = re.compile(rf"({_LETTER}*)({_NUMBER})({_LETTER}*)|({_LETTER}+)")


def tokenize(text):
    """Raw text of ONE sentence -> list of tokens.
    Punctuation is dropped, numbers become <NUM>, and letters attached to a number are kept
    (19ኛ -> <NUM>ኛ, በ1980 -> በ<NUM>). The same number twice in a row is kept once."""
    tokens = []
    prev_number = None                                # raw text of the previous token if it was a bare number
    for m in TOKEN_RE.finditer(normalize(text)):
        pre, number, suf, word = m.groups()
        if word is not None:
            tokens.append(word)
            prev_number = None
            continue
        bare = not pre and not suf
        if bare and number != NUM and number == prev_number:
            continue                                  # identical consecutive number: collapse
        tokens.append(f"{pre}{NUM}{suf}")
        prev_number = number if bare else None
    return tokens


def split_sentences(text):
    """Split on the sentence delimiters and drop empty pieces."""
    return [p.strip() for p in SPLIT_RE.split(text) if p.strip()]


# ------------------------------------------------------------------ deterministic split
def sentence_key(tokens):
    """What identifies a sentence: its normalized tokens joined by spaces."""
    return " ".join(tokens)


def hash64(seed, text):
    """Stable 64-bit hash: the same on every machine and every run (Python's hash() is not)."""
    digest = hashlib.blake2b(f"{seed}|{text}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def split_of_hash(h):
    r = h / 2**64
    return "train" if r < 0.8 else ("dev" if r < 0.9 else "test")


def assign_split(key):
    """80/10/10. The same key always gets the same split."""
    return split_of_hash(hash64(SPLIT_SEED, key))
