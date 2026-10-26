# Extrinsic Evaluation of the Amharic N-gram Language Models (Task 3)

## 1. Goal

Extrinsic evaluation measures how useful a language model is inside a downstream task, whereas intrinsic evaluation (perplexity, Task 2) measures how well the model fits held-out text. We evaluate the n-gram models (n = 1, 2, 3, 4) on two tasks:

1. **Next-word prediction**: the use case behind the predictive-text system of Task 4.
2. **Real vs. shuffled sentence discrimination**: does the model prefer a natural word order over a scrambled one?

## 2. Setup

- **Model:** sparse count-based n-gram model, vocabulary 50,000 words, orders 1-4.
- **Test data:** 489,374 test sentences of at least 5 tokens (including `<s>` and `</s>`).
- **Samples (random, seed 42):**
  - next-word prediction: 2,000 sentences (26,173 evaluated positions)
  - discrimination: 1,000 sentences (998 usable; two were skipped because they could not be shuffled into a different order)
- **Next-word prediction (orders 1-4):** for each position the model sees the previous n-1 words and ranks candidate next words by observed count. This is an exact-order lookup with no smoothing or backoff, so each order can be compared on its own. If the context was never seen in training, the model has no suggestion and the position counts as a miss. This evaluation is separate from the Task 4 application, which combines orders with backoff (see Section 3.4).
- **Excluded positions:** positions whose target is `<UNK>` (4,189 of 32,362, i.e. 12.9%) or `</s>` (2,000, i.e. 6.2%) are excluded, 6,189 in total (19.1%), because the predictor never suggests them. All orders are scored on exactly the same 26,173 positions.
- **Discrimination:** each sentence and a random shuffle of its words are scored by backoff-smoothed log-probability (with `</s>`). The model is correct if the real sentence scores higher.

## 3. Results

### 3.1 Next-word prediction (2,000 sentences, 26,173 positions)

| n | top-1 | top-3 | top-5 | context seen | has suggestion | top-1 (given seen) | top-5 (given seen) |
|---|-------|-------|-------|--------------|----------------|--------------------|--------------------|
| 1 | 1.75% | 4.45% | 5.46% | 100.0% | 100.0% | 1.75% | 5.46% |
| 2 | 10.80% | 18.19% | 22.08% | 100.0% | 100.0% | 10.80% | 22.08% |
| 3 | 13.17% | 19.66% | 22.50% | 86.2% | 85.5% | 15.27% | 26.09% |
| 4 | 11.42% | 14.98% | 16.30% | 55.3% | 54.2% | 20.65% | 29.46% |

"Context seen" is the share of positions whose n-1 word context occurred in training. "Has suggestion" is the share of positions where the model returned at least one ordinary-word candidate; it is slightly lower than "context seen" because some seen contexts were only ever followed by `<UNK>` or `</s>`. The "given seen" columns divide the number of hits by the number of positions with a seen context. Top-1, top-3 and top-5 in the first three columns use all 26,173 positions, so a position with an unseen context counts as a miss.

### 3.2 Real vs. shuffled discrimination (998 sentences)

| n | accuracy (ties = ½) | strict accuracy (ties = 0) | wins | ties |
|---|---------------------|----------------------------|------|------|
| 1 | 0.500 | 0.000 | 0 | 998 |
| 2 | 0.989 | 0.989 | 987 | 0 |
| 3 | 0.987 | 0.987 | 985 | 0 |
| 4 | 0.989 | 0.989 | 987 | 0 |

### 3.3 Worked examples on real Amharic sentences

Four sentences were checked: two typed by us (the first two) and two random test sentences. Log-probabilities are natural-log values for the whole sentence (higher is more likely); shuffles are random reorderings of the same words.

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

### 3.4 Next-word prediction with the backoff predictor (small subsample)

The Task 4 application ranks the whole vocabulary by backoff conditional probability using the model's `predict_next()` method. This is slow (roughly 2 seconds per call), so it was evaluated only on the first 11 sentences of the next-word sample: 197 positions with 185 distinct contexts. The exact-order models were re-scored on the same 197 positions, with the same exclusions, so the five rows are comparable with each other but **not** with Section 3.1.

| predictor | top-1 | top-3 | top-5 | has suggestion |
|---|-------|-------|-------|----------------|
| n=1 (exact) | 1.52% | 5.58% | 5.58% | 100.0% |
| n=2 (exact) | 11.68% | 19.80% | 23.35% | 100.0% |
| n=3 (exact) | 10.66% | 19.80% | 21.32% | 82.7% |
| n=4 (exact) | 10.66% | 14.21% | 14.72% | 51.8% |
| backoff (`predict_next`) | 14.72% | 22.84% | 25.89% | 100.0% |

Only the last three words of the context are passed to `predict_next()`, and the `<s>` padding is dropped, as in the application; at the first word of a sentence the backoff predictor therefore sees an empty context and returns unigram suggestions, whereas the exact-order models see `<s>` padding. The backoff row was checked to return exactly the same suggestions as a direct call to `predict_next()`.

## 4. Analysis

- **Context matters most at the first step.** Going from the unigram to the bigram raises top-1 accuracy from 1.7% to 10.8%. The unigram model suggests the same frequent words everywhere, so its score is the no-context baseline.
- **Accuracy and coverage move in opposite directions.** When a 4-gram context was seen in training, the model is the most accurate (20.6% top-1, 29.5% top-5), but about 44.7% of positions in the test sample had a 4-gram context that never occurred in training (13.8% for trigrams). The trigram model gives the best overall top-1 (13.2%) because it balances the two. At top-5 the trigram (22.5%) and bigram (22.1%) are essentially level, so we do not claim a difference there. The "given seen" accuracies are computed on different subsets of positions for each order (positions with a seen long context are mostly common phrases, which are easier), so they should not be read as a like-for-like comparison of the orders. The worked examples show the same sparsity effect: the long test sentence hit "(context unseen)" several times, and the 4-gram model missed every word in it.
- **Sparsity, not model logic, limits the higher orders.** A deployed system should back off to shorter contexts instead of returning no suggestion. This motivates the backoff predictor in Task 4.
- **The backoff predictor scored highest on the small subsample, but the sample is too small to prove it.** On 197 positions it reached 14.7% top-1 and 25.9% top-5 against 11.7% and 23.4% for the best exact order (bigram), and it is the only predictor besides the unigram that always returns suggestions. With 197 positions (from 11 sentences), the 95% interval on a top-5 accuracy near 25% is roughly ±6 percentage points, and positions within a sentence are not independent, so the gap is within sampling noise. We therefore report this as consistent with the benefit of backoff, mainly full coverage, not as a demonstrated accuracy gain.
- **Discrimination is saturated at bigrams.** Orders 2-4 differ by at most two sentences out of 998 (987, 985 and 987 wins), which is within sampling noise, so we do not claim any ordering between them. The unigram model scores exactly 0.5 (all ties) because it assigns the same probability to any ordering of the same words.
- **Trigram and 4-gram scores on shuffled text are identical in the examples.** The backoff method falls back to the shorter context whenever a context is unseen, and shuffled word sequences almost never form seen 4-gram contexts, so n=4 reduces to the n=3 probability. This is a property of the smoothing method, not a measurement of the 4-gram model's quality.
- **Suggestions are often reasonable even when they miss.** For "አዲስ አበባ", the model suggests ላይ, ውስጥ, ከተማ (all natural continuations) while the actual word was የኢትዮጵያ. Exact-match accuracy therefore understates usefulness. Amharic's rich morphology also hurts: the form ትገኛለች was not suggested after "ኢትዮጵያ በአፍሪካ ቀንድ".

## 5. Conclusion

Trigrams give the best overall top-1 next-word accuracy (13.2%) and are level with bigrams at top-5; 4-grams are the most accurate when their context has been seen (20.6% top-1) but that is the case for only about 55% of positions; and bigrams already capture nearly all of the word-order signal in the discrimination task. The models behave as expected for count-based n-grams: they capture local word order and common phrases, but sparsity limits the higher orders.

## Result files

- `results/extrinsic_evaluation/extrinsic_next_word.csv`: next-word results for orders 1-4 (Section 3.1)
- `results/extrinsic_evaluation/extrinsic_discrimination.csv`: discrimination results (Section 3.2)
