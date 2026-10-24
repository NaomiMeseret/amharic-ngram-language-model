# Intrinsic Evaluation and Text Generation (Task 2 and Task 1.8)

Amharic n-gram language model, NLP Assignment 1.

## Setup

We evaluated the full model (trained on all 4,067,589 training sentences, vocabulary of 50,000 entries) on the test set of 504,690 sentences, which contains 8,074,537 prediction events (every word, including `<UNK>`, plus the end-of-sentence marker). Perplexity is the exponential of the negative mean log probability over these events, so lower is better. For an n-gram model of order n, every sentence is padded with n−1 start markers, as in training. The test set was not used for any tuning. MLE and add-k were computed on the complete test set; backoff and interpolation are much slower in our implementation, so they were computed on a random sample of 300 sentences drawn from the test set.

We compared four estimation methods:

- **Unsmoothed counts (MLE).** Probabilities are taken directly from the counts.
- **Add-k smoothing.** A small constant k is added to every count. The best k for each order was chosen on a sample of 5,000 development sentences: k = 0.00001 for n=1, 0.01 for n=2, 0.001 for n=3 and 0.0003 for n=4. Add-1 is shown for reference.
- **Absolute-discount backoff.** A fixed discount of 0.5 is subtracted from observed counts, and the freed probability goes to the next shorter context.
- **Linear interpolation** of the four orders, with weights 0.2, 0.4, 0.1 and 0.3 for the unigram to 4-gram models, tuned on the development set. It combines all four orders, so it gives one value, not one per order.

## Perplexity results

| n | MLE | Test n-grams never seen in training | Add-k | Add-1 | Backoff (300 sentences) | Interpolation (300 sentences) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2,428 | 0% | 2,428 | 2,428 | 2,392 | - |
| 2 | infinite | 13.6% | 538 | 1,490 | 463 | - |
| 3 | infinite | 45.7% | 1,910 | 9,517 | 396 | - |
| 4 | infinite | 67.4% | 6,483 | about 21,600 | 385 | 309 (all four orders) |

The last two columns use a smaller sample than the others, so they are not strictly comparable. For the unigram model the two samples give close results (2,392 on the sample and 2,428 on the whole test set), which suggests that the sample is reasonably representative.

## Discussion of the results

With unsmoothed counts, only the unigram model has a finite perplexity. For the bigram, trigram and 4-gram models, 13.6%, 45.7% and 67.4% of the test n-grams never occurred in training. They receive probability zero, which makes the perplexity infinite. This is the sparsity problem: the longer the context, the less likely it is that the same sequence appeared in training.

Smoothing removes the zero probabilities, but the choice of method matters a great deal. With add-k, perplexity improves from the unigram (2,428) to the bigram model (538) and then gets worse again for the trigram (1,910) and 4-gram (6,483) models. Add-k gives every unseen continuation of a context the same small probability, spread evenly over about 50,000 words. This is a poor guess when most long contexts are unseen, so the longer models are penalized heavily. Add-1 is much worse still (about 21,600 for n=4), because it moves too much probability to words that never follow the context.

Backoff and interpolation estimate an unseen long context from shorter contexts instead. With backoff, perplexity decreases steadily as n grows (2,392, 463, 396, 385), so the longer contexts do help once unseen sequences are handled sensibly. Linear interpolation of all four orders gave the lowest perplexity of all methods, 309. The 4-gram model scores 6,483 with add-k and 385 with backoff (on different samples), so on our measurements the smoothing method matters more than the model order.

## Random sentence generation

We generated sentences from each model by sampling one word at a time from the observed counts after the current context, starting from the sentence-start marker and stopping at the end marker or after 30 words (sentences cut off at 30 words end with "..."). `<UNK>` was excluded from sampling. When a context had never been observed, the generator used a shorter context for that step. In a run of ten sentences per model (random seed 7), this was needed in 0% of the steps for n=1 and n=2, 5% for n=3 and 9% for n=4. The token `<NUM>`, which stands for a number in the corpus, can appear in the output.

The output changes clearly with n:

- **n=1:** The sentences are random strings of frequent words with no relation between neighbors, for example አሳታፊ ነጋዴው ላፊ ቤቱ ብለዋል ጥያቄ ባሻገር. One of the ten samples is empty, because the end marker was drawn first, and one was cut off at 30 words.
- **n=2:** Neighboring words fit together, and short sentences such as ኢትዮጵያዊት ነኝ are well formed. Longer sentences lose their thread after a few words and drift from one topic to another.
- **n=3:** Longer stretches read like natural Amharic phrases, for example ነገር ግን የአካባቢው አስተዳዳሪዎች የሀይማኖት መሪዎችና ፖለቲከኞች, but the overall meaning of a long sentence is often not coherent, and one of the ten samples was cut off at 30 words.
- **n=4:** Most sentences are locally fluent and several are complete and read as grammatical Amharic, for example ኢኮኖሚውን ማስተካከል ይገባል ብለዋል, በመኢአድ ውስጥ በተፈጠረ ውዝግብ ከመኢአድ አመራር እንዲለቅ ተደርጎ ነበር and ጉባኤው በተለያዩ ጊዜያት በሚያወጣቸው ሪፖርቶች በአገሪቱ በሚገኙ አንዳንድ አካባቢዎች አሁንም ችግሮቹ በስፋት ይስተዋላሉ. Some long sentences still jump between topics (for example በዚያም ቀን እንዲህ ይሆናል ስጋ ኢትዮጵያ ውስጥ ብቻ ሳይሆን በሚኒስትሮች ምክር ቤትም እንዲሁ), and the broken abbreviation ዶ ር appears in one sentence.

In short, as n increases the model has more context and produces text that is more grammatical and more natural. However, a longer context has been seen fewer times in training, so the model has fewer observed continuations to choose from. Higher-order models therefore copy exact phrases from the training text more often and must fall back to a shorter context more often. By construction, at every step that did not need a shorter context, the n consecutive words ending at that step (counting the start marker) occur in the training text, so about 91% of the steps of the 4-gram model reproduce a 4-word sequence from the corpus. We also searched the training data for each complete generated sentence (sentences cut off at 30 words were skipped): none of the 10 bigram sentences and none of the 9 trigram sentences, but 2 of the 10 4-gram sentences, appear word for word in the training data. The other 4-gram sentences are new combinations of word sequences that do occur in the corpus. The non-empty sentences had an average length of 10.0 words (n=1), 9.8 (n=2), 14.4 (n=3) and 11.9 (n=4); ten samples per model are too few to show a trend in length.

## Limitations

- The backoff and interpolation perplexities were measured on 300 test sentences, not the whole test set.
- Perplexity treats `<UNK>` as an ordinary word, and about 14% of the test tokens are `<UNK>` (14.06%). This makes the perplexity look lower than it would be for a model with a larger vocabulary.
- About 24% of the test sentences are duplicates (120,510 of 504,690), and the test set was not de-duplicated, so repeated sentences count several times.
- The add-k constants and interpolation weights were tuned on a 5,000-sentence development sample, and we did not re-tune them for any other model.
- The comments on sentence quality come from our own reading of a small number of generated sentences (ten per model). They were not rated by a panel of native speakers.
- Generated sentences can contain names of real people and statements taken from the corpus (for example in the n=4 output), so they should not be read as factual.
- We did not evaluate the model trained without stopwords, because the assignment does not require it.
