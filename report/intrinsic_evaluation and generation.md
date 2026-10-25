# Task 1.8: Random Sentence Generation with N-grams (n = 1 to 4)

## 1. Task

Generate random sentences from the unigram, bigram, trigram and 4-gram models, and explain what happens as n increases, based on the generated output.

## 2. Method

- **Model:** the trained sparse count-based n-gram model (vocabulary 50,000, orders 1-4).
- **Sampling:** each sentence starts from the sentence-start context (`<s>`). At every step the next word is drawn at random from the model's observed counts for the current n-1 word context (higher count, higher chance). Generation stops when `</s>` is drawn or after the length limit. A sentence that reaches the limit is shown ending in `...`.
- **No smoothing.** Only continuations seen in training can be generated. `<UNK>` is excluded from the candidates.
- **Fallback.** If the current n-1 word context was never seen, the generator uses a shorter context for that step. The *fallback ratio* is the share of steps that needed this.
- **Two runs:**
  - Examples: 5 sentences per order, maximum 25 words, seed 42.
  - Statistics: 100 sentences per order, maximum 30 words, seed 0.
- **Statistics reported per order** (over the 100 sentences):
  - *ended naturally*: share of sentences that stopped at `</s>` before the limit
  - *avg words*: mean sentence length
  - *fallback %*: share of generation steps that used a shorter context
  - *k-grams in training %*: share of the generated word windows of length 2, 3 and 4 that occur in the training counts (a measure of copying versus novelty; windows at the very start of a sentence are checked without start padding, so the percentages are slightly conservative)

## 3. Results

### 3.1 Examples (generated text, copied from the notebook output)

**n = 1**
- በዴሞክራሲያዊ ታላቅ የበኩር በወር \<NUM\> የዲግሪ የምናገኘው ኢዴፓ እና እድሜ ምላሽ እንዲጫወት ወዲያውኑ በመጡ ው ምክንያት ጠላት
- በደሴ ቀኝ ሊደረግ ሌላው መጓደልና ለኮሮና በስራው ህዝብ እንዳለው

**n = 2**
- ለአካባቢው ነዋሪዎች የፌዴራልና የክልል ከተማ ብራስልስ አውስትራሊያ የሄዱት
- ካህናት አሏት የራሷ ሆነው ይታያሉ

**n = 3**
- ለአካባቢው ቀውስ መነሻ የሆነው ሰው ከህክምናው በተጨማሪ ህሙማን መስራት እንዲችሉ ግንዛቤ እንደፈጠረላቸው ተናግረዋል
- ይህን የዜጎች ብሶት ምን እንደሆነ አላውቅም ወደ ኋላ መለስ ብዬ ሳስብ የባሰ የሚሆነው ኢትዮጵያዊ ከድህነት ወለል በታች እንደሚኖሩ ያሳያል

**n = 4**
- ለአካባቢው ቀውስ መነሻ የሆነው የሰሜን ምስራቅ ናይጄሪያ ክፍልን የተቆጣጠረው ቦኮሀራም ባለፉት \<NUM\> አመታት በፌዴሬሽኑ በኩል በተለያዩ አቅጣጫዎች ስፖርቱን ወደ ቢዝነስ አቅጣጫ ለማስገባት የሚያስችሉ ሙከራዎች ተደርገዋል ...
- ከሁለት አመት በፊት የጸረ አፓርታይድ ታጋይና የአለም የሰላም ኖቤል ሽልማት አሸናፊ ሆኑ
- ብለው ይጠይቃሉ
- ምንድን ነው

### 3.2 Statistics (100 sentences per order)

| n | ended naturally % | avg words | fallback % of steps | bigrams in training % | trigrams in training % | 4-grams in training % |
|---|---|---|---|---|---|---|
| 1 | 86.0 | 12.4 | 0.0 | 28.3 | 0.2 | 0.0 |
| 2 | 89.0 | 13.3 | 0.0 | 100.0 | 17.5 | 1.7 |
| 3 | 88.0 | 13.4 | 2.8 | 100.0 | 96.7 | 41.5 |
| 4 | 89.0 | 14.0 | 7.7 | 100.0 | 97.2 | 90.3 |

For n = 1 the fallback is 0% by definition (there is no context to back off from).

## 4. What happens as n increases

1. **Local coherence rises sharply.**
   - n = 1 produces unordered strings of frequent words: only 28.3% of adjacent word pairs ever occur in the corpus, and practically no trigrams do (0.2%). The output has no grammar.
   - n = 2 guarantees that every adjacent pair is a real training pair (100%), so short stretches look natural (for example "የፌዴራልና የክልል", "ሆነው ይታያሉ"), but the sentence wanders because the model remembers only one word. Only 17.5% of its trigrams occur in training.
   - n = 3 produces complete clauses: 96.7% of its trigrams are real, and sentences such as "ይህን የዜጎች ብሶት ምን እንደሆነ አላውቅም ወደ ኋላ መለስ ብዬ ሳስብ ..." read like natural Amharic. Longer-range structure is still missing, since 41.5% of its 4-grams are new combinations.
   - n = 4 reads the most fluently, with sentences that look like news text (for example the sentence about Boko Haram).

2. **Originality falls as copying rises.** The share of generated 4-grams that appear verbatim in the training data goes from 0.0% (n = 1) and 1.7% (n = 2) to 41.5% (n = 3) and 90.3% (n = 4). A 4-gram model mostly stitches together phrases it has seen. Evidence in the examples: the n = 3 and n = 4 sentences that start with "ለአካባቢው ቀውስ መነሻ የሆነው" share their first words (and the two sentences starting "ይህን የዜጎች ብሶት ምን" likewise), which indicates that after such rare prefixes the higher-order models have only a few continuations to choose from. The same random seed also gives the same first word at n = 2, 3 and 4, so shared first words alone prove nothing; the shared multi-word prefixes are the informative part.

3. **Diversity and robustness fall.** Several n = 4 outputs are short, formulaic sentences ("ብለው ይጠይቃሉ", "ምንድን ነው"). At the same time the fallback ratio rises from 0% (n = 2) to 2.8% (n = 3) and 7.7% (n = 4): with longer contexts the generator more often reaches a context it has never seen and has to shorten it. This is consistent with the sparsity measured in the perplexity evaluation, where the share of unseen test n-grams was 13.6% (bigram), 45.7% (trigram) and 67.4% (4-gram).

4. **Sentence length and endings barely change.** The average length grows slightly (12.4 to 14.0 words) and 86-89% of sentences end naturally at every order. With 100 sentences per order, differences of a few points are within sampling noise, so we draw no conclusion from them.

5. **Fluent is not the same as meaningful.** Even the n = 4 text is only a chain of locally plausible phrases. For example, "ከሁለት አመት በፊት የጸረ አፓርታይድ ታጋይና የአለም የሰላም ኖቤል ሽልማት አሸናፊ ሆኑ" is grammatical, but the model has no notion of what is true or consistent across a sentence, and a longer sentence can drift to another topic.

**Summary:** increasing n trades novelty for fluency. Low n gives creative but incoherent text; high n gives coherent text that is largely copied from the training data, and it needs far more data (or backoff) to avoid unseen contexts. A bigram or trigram gives the best balance for generating varied text, while the 4-gram output is the most readable but the least original.

# Task 2: Intrinsic Evaluation of the Amharic N-gram Models (Perplexity)

## 1. What we did

We trained four n-gram language models: unigram, bigram, trigram and 4-gram (n = 1, 2, 3, 4). In this task we check how good each model is by showing it text it has never seen (the test file) and measuring how well it predicts that text. The measure we use is **perplexity**. This is called intrinsic evaluation because we judge the model directly, without using it inside another task (that is Task 3).

## 2. What is perplexity?

Perplexity tells us how "surprised" a model is by the test text.

- The model gives a probability to every word in the test text, based on the words before it.
- If the model gave high probabilities to the real words, it is not surprised, so perplexity is low.
- If it gave tiny probabilities to the real words, it is very surprised, so perplexity is high.

Formula:

    Perplexity = exp( -(1/N) * sum of log P(word | previous words) )

N is the number of predictions. We use logs because multiplying millions of small probabilities would become zero on a computer, while adding their logs works fine.

A simple way to read it: a perplexity of 400 means the model is as unsure as if it had to pick the next word from 400 equally likely words. **Lower is better.**

## 3. Data and settings

- **Model:** vocabulary of 50,000 entries (49,998 of them can be predicted), orders 1 to 4.
- **Test data:** `test.npy`, 8,579,227 tokens. We count a prediction for every word and for the end-of-sentence marker `</s>`, which gives 8,074,537 predictions. The start marker `<s>` is only used as context.
- **Unknown words:** 12.40% of the test tokens are `<UNK>` (words that are not in the 50,000-word vocabulary).
- **Settings were chosen on a separate dev set, not on the test set.** The best add-k values were k = 0.00001, 0.01, 0.001 and 0.0003 for n = 1 to 4. The best mixing weights for interpolation were 0.2, 0.4, 0.1, 0.3 (unigram, bigram, trigram, 4-gram).

## 4. The five ways we compute probabilities

1. **MLE (no smoothing):** probability = how often the n-gram appeared / how often its first words appeared. If the n-gram never appeared in training, the probability is 0, and perplexity becomes infinite.
2. **Add-1 (Laplace):** pretend every possible word was seen one extra time. This removes zeros, but with 50,000 words it gives too much probability to words that never appear.
3. **Add-k:** the same idea with a small number k instead of 1. We chose k on the dev set.
4. **Backoff:** if the long pattern was never seen, use a shorter pattern instead. We take a little probability (0.5 of each count) from the seen words and give it to the others using the shorter context.
5. **Interpolation:** always mix the four models: 0.2 x unigram + 0.4 x bigram + 0.1 x trigram + 0.3 x 4-gram. A missing 4-gram then never leaves the model empty-handed.

## 5. Results

### 5.1 Whole test set (8,074,537 predictions)

| n | MLE | predictions with zero MLE probability | add-1 | add-k (best k) | k |
|---|-----|------|-------|----------------|---|
| 1 | 2,428 | 0.00% | 2,428 | 2,428 | 0.00001 |
| 2 | infinite | 13.58% | 1,490 | **538** | 0.01 |
| 3 | infinite | 45.66% | 9,517 | 1,910 | 0.001 |
| 4 | infinite | 67.37% | 21,603 | 6,483 | 0.0003 |

### 5.2 Same 500 test sentences, all methods (7,938 predictions)

Backoff and interpolation are slow to compute, so we used 500 random whole sentences from the test set. We also recomputed add-k on the same 500 sentences so the comparison is fair.

| n | add-k | backoff | interpolation |
|---|-------|---------|---------------|
| 1 | 2,378 | 2,378 | not computed |
| 2 | 533 | 458 | not computed |
| 3 | 1,961 | **409** | not computed |
| 4 | 6,741 | 410 | **334** |

For n = 1, backoff is the same as add-k because there is no shorter pattern to go back to. Interpolation was computed only for n = 4, because the tuned weights are for the mix of all four models.

A chart of this table is saved as `results/intrinsic_ppl_by_order.png` (log scale).

### 5.3 Comparing n = 1, 2, 3 and 4

How much perplexity changes when we add one more word of context (a minus sign means better):

| Step | add-k (whole test set) | add-k (500 sentences) | backoff (500 sentences) |
|---|---|---|---|
| unigram to bigram | -77.8% | -77.6% | -80.7% |
| bigram to trigram | +254.8% | +268.2% | -10.7% |
| trigram to 4-gram | +239.5% | +243.7% | +0.5% |

On the same 500 sentences, interpolation (334) is 18.7% lower than backoff (410) and 37.4% lower than the best add-k result (the bigram, 533).

### 5.4 Examples on real Amharic test sentences

We picked three random test sentences (9 to 14 words, no unknown words) and measured the perplexity of each one. "Bits" below are the same idea as perplexity, written as -log2 of the probability: 3 bits means the word was as surprising as one out of 8 equally likely words.

| Sentence | bigram add-k | 4-gram add-k | 4-gram backoff | 4-gram interpolation |
|---|---|---|---|---|
| በስፖርቱ ያሳለፉ መልሰው ስፖርቱን እንዲያገለግሉ የማብቃት ስራ አልሰራም | **4,684** | 61,022 | 8,610 | 7,176 |
| በዚህም በአካባቢው የሚኖሩ \<NUM\> ሺህ የሚደርሱ ነዋሪዎች ተጠቃሚ እንደሚሆኑ ነው የተገለጸው | 144 | 1,077 | **60** | 70 |
| በመላክ ከራሱ አልፎ ለበርካቶች የስራ እድል ፈጥሯል አስራ ሁለቱም ከእሱ ጋር ነበሩ | 1,059 | 10,812 | 115 | **110** |

What we saw:

- Many words in the 4-gram add-k column cost exactly 15.6 bits. That is log2(49,998), which means the model gave up and guessed evenly among all words, because it had never seen the 3 words before.
- In sentence 3, the word ነበሩ cost 12.0 bits for the bigram but only 0.2 bits for 4-gram backoff. The phrase "ሁለቱም ከእሱ ጋር" is almost always followed by ነበሩ in the training text, so the longer context helped a lot.
- Sentence 1 has many rare words, and there the plain bigram won. Longer contexts cannot help when the words are rare. So the best method is not best for every sentence, only on average.

## 6. What the results mean

1. **One word of context helps a lot.** Going from unigram to bigram lowers perplexity by about 78% (add-k) or 81% (backoff).
2. **After the bigram, it depends on how we handle unseen patterns.** With add-k, more context makes things much worse (538, 1,910, 6,483). With backoff, the trigram is a little better than the bigram and the 4-gram is the same as the trigram (409 and 410; this difference is too small to say one is better).
3. **The reason is sparsity.** Many longer patterns in the test data never appeared in training: 13.6% of bigrams, 45.7% of trigrams and 67.4% of 4-grams. MLE gives them probability 0 (infinite perplexity). Add-k gives them a nearly even guess over about 50,000 words, which is a bad guess. Backoff and interpolation use shorter patterns instead.
4. **Add-1 is a poor choice for this data.** It was 2.8 times worse than tuned add-k for n = 2 and 3.3 times worse for n = 4.
5. **Interpolation was the best overall (334).** It always uses all four models, so it uses the extra context when it exists and the shorter patterns when it does not.
6. **Dev and test agree.** The perplexities we expected from the dev set (about 2,443, 526, 1,841 and 6,183 for add-k) are within about 5% of the test results, so tuning on dev did not overfit.

## 7. Conclusion

All four models were evaluated on the same unseen test text. Perplexity falls sharply from the unigram (about 2,400) to the bigram (about 500), and it does not keep falling for longer n unless the model can fall back to shorter patterns. Plain add-k gets worse for the trigram and 4-gram, backoff fixes this, and interpolation of all four models gives the best result (334). 