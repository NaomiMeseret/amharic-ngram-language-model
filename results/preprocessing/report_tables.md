### Table 1: Corpus after preprocessing
| Split | Sentences | Tokens | Avg. length | `<UNK>` % | `<NUM>` tokens % |
|---|---:|---:|---:|---:|---:|
| train | 4,067,589 | 60,807,436 | 14.9 | 13.94 | 1.86 |
| dev | 508,114 | 7,589,622 | 14.9 | 14.07 | 1.84 |
| test | 504,690 | 7,569,847 | 15.0 | 14.06 | 1.85 |

### Table 2: Share of train tokens covered by the top-k vocabulary words
| Top-k words | Coverage % |
|---:|---:|
| 100 | 18.47 |
| 1,000 | 42.20 |
| 5,000 | 62.28 |
| 10,000 | 70.58 |
| 20,000 | 78.02 |
| 49,996 | 86.06 |

### Table 3: 20 most frequent tokens (train)
| Rank | Token | Count |
|---:|---|---:|
| 1 | ነው | 921,688 |
| 2 | <NUM> | 887,452 |
| 3 | ላይ | 594,173 |
| 4 | ውስጥ | 273,656 |
| 5 | ወደ | 260,934 |
| 6 | እና | 246,281 |
| 7 | ጋር | 229,818 |
| 8 | ነበር | 217,891 |
| 9 | ግን | 214,337 |
| 10 | ጊዜ | 193,009 |
| 11 | ነገር | 171,940 |
| 12 | ደግሞ | 165,689 |
| 13 | አንድ | 152,404 |
| 14 | ቤት | 142,527 |
| 15 | ይህ | 141,351 |
| 16 | ብቻ | 133,773 |
| 17 | እንደ | 133,026 |
| 18 | መንግስት | 131,776 |
| 19 | አቶ | 131,651 |
| 20 | ስራ | 124,806 |

### Table 4: Sentence length (words, after cleaning)
| Mean | Median | 90th pct | 99th pct | Max |
|---:|---:|---:|---:|---:|
| 15.0 | 12 | 28 | 53 | 100 |

### Table 5: Preprocessing outcome
| Item | Value |
|---|---:|
| Rules version | preprocessing-v2 |
| Sample fraction | 1.0 |
| Sample seed / split seed | 42 / 7 |
| Sentence pieces in the corpus | 5,089,654 |
| Words in the raw corpus (whitespace-separated) | 79,375,985 |
| Sentences sampled | 5,089,654 |
| Sentences dropped (over 100 tokens) | 7,673 |
| Tokens discarded by that cutoff | 1,385,403 (1.79%) |
| Empty after cleaning | 1,588 |
| Repeated sentences (train / dev / test) | 988,589 / 123,668 / 120,510 |
| Vocabulary size | 50,000 |
| Unique words (train / all splits) | 1,581,903 / 1,782,656 |
| `<UNK>` rate % (train / dev / test) | 13.94 / 14.07 / 14.06 |

### Table 6: Before and after normalization
| Raw | Tokens |
|---|---|
| ሲጠጣ፣ በሩቅ እያየች | ሲጠጣ በሩቅ እያየች |
| የአለም የእግር ኳስ ዋ ለ19ኛ ጊዜ | የአለም የእግር ኳስ ዋ ለ<NUM>ኛ ጊዜ |
| በ1980 ዓ.ም | በ<NUM> አ ም |
| መጽሐፍ ሐምሌ ዓለም ሠላም | መጽሀፍ ሀምሌ አለም ሰላም |
| ሃገር ሓገር ኃይል ሀገር | ሀገር ሀገር ሀይል ሀገር |