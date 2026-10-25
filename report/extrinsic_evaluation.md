# Extrinsic Evaluation of the Amharic N-gram Language Models (Task 3)

## 1. Goal

Extrinsic evaluation instead how useful the model is inside a downstream task. We evaluate the n-gram models (n = 1, 2, 3, 4) on two tasks:

1. **Next-word prediction**: the use case behind the predictive-text system of Task 4.
2. **Real vs. shuffled sentence discrimination**: does the model prefer a natural word order over a scrambled one?

## 2. Setup

- **Model:** sparse count-based n-gram model, vocabulary 50,000 words, orders 1-4.
- **Training corpus:** [TODO: size and split procedure, from reports/training_report.json]
- **Test data:** 489,374 test sentences of at least 5 tokens (including `<s>` and `</s>`).
- **Samples (random, seed 42):**
  - next-word prediction: 2,000 sentences (26,173 evaluated positions)
  - discrimination: 1,000 sentences (998 usable; two could not be shuffled into a different order)
- **Next-word prediction:** for each position the model sees the previous n-1 words and ranks candidate next words by observed count. Exact-order lookup, no smoothing or backoff. If the context was never seen in training, the position counts as a miss. Positions whose target is `<UNK>` or `</s>` (6,189 of 32,362, i.e. 19.1%) are excluded, because the predictor never suggests them.
- **Discrimination:** each sentence and a random shuffle of its words are scored by backoff-smoothed log-probability (with `</s>`). The model is correct if the real sentence scores higher. A tie counts as half a point.

## 3. Results

### 3.1 Next-word prediction

| n | top-1 | top-3 | top-5 | coverage | top-1 (context seen) | top-5 (context seen) |
|---|-------|-------|-------|----------|----------------------|----------------------|
| 1 | 1.75% | 4.45% | 5.46% | 100.0% | 1.75% | 5.46% |
| 2 | 10.80% | 18.19% | 22.08% | 100.0% | 10.80% | 22.08% |
| 3 | 13.17% | 19.66% | 22.50% | 85.5% | 15.41% | 26.33% |
| 4 | 11.42% | 14.98% | 16.30% | 54.2% | 21.10% | 30.11% |

Coverage is the share of positions whose n-1 word context occurred in training. The "context seen" columns are derived by dividing by coverage.

### 3.2 Real vs. shuffled discrimination (998 sentences)

| n | accuracy | ties |
|---|----------|------|
| 1 | 0.500 | 998 |
| 2 | 0.989 | 0 |
| 3 | 0.987 | 0 |
| 4 | 0.989 | 0 |

### 3.3 Worked examples on real Amharic sentences

Four sentences were checked.Log-probabilities are natural-log values for the whole sentence (higher is more likely); shuffles are random reorderings of the same words.

| Sentence | Shuffled version | log P real / shuffled (n=2) | (n=3) | (n=4) |
|---|---|---|---|---|
| ኢትዮጵያ በአፍሪካ ቀንድ ትገኛለች | ትገኛለች ኢትዮጵያ ቀንድ በአፍሪካ | -25.9 / -55.8 | -25.1 / -57.2 | -25.6 / -57.2 |
| አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት | ከተማ ናት አበባ ዋና አዲስ የኢትዮጵያ | -29.0 / -58.7 | -19.5 / -65.6 | -16.3 / -65.6 |
| በመድረኩ ከለውጥ በኋላ የብልጽግና ጉዞ ጅማሮ እንዴት እየሄደ ነው | በኋላ ጉዞ ከለውጥ ጅማሮ ነው እየሄደ በመድረኩ የብልጽግና እንዴት | -69.7 / -94.7 | -64.1 / -102.2 | -63.1 / -102.2 |
| የአምላክ መንግስት ተገዢዎች አስደሳችና ፍሬያማ ስራ ይኖራቸዋል | ይኖራቸዋል የአምላክ ፍሬያማ መንግስት ተገዢዎች አስደሳችና ስራ | -53.1 / -81.8 | -50.6 / -84.1 | -52.7 / -84.1 |

The real sentence won in all 12 comparisons.

Next-word suggestions (exact n-gram, top 5), selected positions from the same sentences:

| Input so far | Actual next | n=3 suggestions | Hit (n=2 / 3 / 4) |
|---|---|---|---|
| አዲስ | አበባ | አበባ, ዘመን, አበባን, አድማስ, ነገር | ✓ / ✓ / ✓ |
| አዲስ አበባ | የኢትዮጵያ | ላይ, ውስጥ, ከተማ, ዩኒቨርሲቲ, ጥር | ✗ / ✗ / ✗ |
| አዲስ አበባ የኢትዮጵያ ዋና | ከተማ | ከተማ, ዋና, ስራ, ዳይሬክተር, ፖለቲካ | ✓ / ✓ / ✓ |
| ኢትዮጵያ በአፍሪካ ቀንድ | ትገኛለች | አካባቢ, ላይ, ውስጥ, ሰላምና, ሰላም | ✗ / ✗ / ✗ |
| በመድረኩ ከለውጥ | በኋላ | (context unseen) | ✓ / ✗ / ✗ |
| በመድረኩ ከለውጥ በኋላ የብልጽግና | ጉዞ | ፓርቲ | ✓ / ✗ / ✗ |
| የአምላክ መንግስት ተገዢዎች አስደሳችና | ፍሬያማ | (context unseen) | ✗ / ✗ / ✗ |

Over all 22 positions in the four sentences, the actual word was in the top 5 for 7 positions at n=2, 7 at n=3 and 6 at n=4. This is anecdotal (22 positions, two of the sentences chosen by us) and is not a substitute for Section 3.1.

## 4. Analysis

- **Context matters most at the first step.** Going from the unigram to the bigram raises top-1 accuracy from 1.7% to 10.8%. The unigram model suggests the same frequent words everywhere, so its score is the no-context baseline.
- **Accuracy and coverage move in opposite directions.** When a 4-gram context was seen in training, the model is the most accurate (21.1% top-1), but 45.8% of 4-gram contexts in the test data never occurred in training. The trigram model gives the best overall top-1 (13.2%) because it balances the two. The worked examples show the same effect: the long test sentence hit "(context unseen)" several times, and the 4-gram model missed every word in it.
- **Sparsity, not model logic, limits the higher orders.** A deployed system should back off to shorter contexts instead of returning no suggestion. This motivates the backoff predictor in Task 4.
- **Trigram and 4-gram scores on shuffled text are identical in the examples.** The backoff method falls back to the shorter context whenever a context is unseen, and shuffled word sequences almost never form seen 4-gram contexts, so n=4 reduces to the n=3 probability. This is a property of the smoothing method.
- **Suggestions are often reasonable even when they miss.** For "አዲስ አበባ", the model suggests ላይ, ውስጥ, ከተማ (all natural continuations) while the actual word was የኢትዮጵያ. Exact-match accuracy therefore understates usefulness.

## 5. Limitations

- Absolute accuracy is modest (top-5 of 22-30% at best). Word-level n-grams over a 50,000-word vocabulary are sparse for a morphologically rich language, and many misses are plausible alternatives.
- Out-of-vocabulary words: 19.1% of positions had an `<UNK>` or `</s>` target and were excluded, so accuracy applies to in-vocabulary words only.
- Exact-order evaluation is a lower bound for a backoff system. A backoff-based next-word evaluation was not run.
- Shuffled sentences are an easy negative; real grammatical errors are subtler, so this task cannot rank orders 2-4.
- Results use samples of 2,000 and 1,000 sentences; no confidence intervals were computed.
- The worked examples (4 sentences) are illustrative only. We did not check whether the typed sentences occur in the training data.

## 6. Conclusion

Trigrams give the best overall next-word accuracy, 4-grams the best accuracy when their context has been seen, and bigrams already capture most of the word-order signal. The models behave as expected for count-based n-grams: they capture local word order and common phrases, but sparsity limits the higher orders.
