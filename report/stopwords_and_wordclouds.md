# Stopword Removal and Word Clouds (Tasks 1.5 and 1.6)

Amharic n-gram language model, NLP Assignment 1.

## Stopword list

We compiled a list of 86 Amharic stopwords by going through the 200 most frequent words of the vocabulary and keeping the function words, that is, words with a grammatical rather than a lexical role. The list contains copulas and auxiliaries (ነው, ነበር, ናቸው), prepositions and postpositions (ወደ, ስለ, ላይ, ውስጥ, ጋር), conjunctions and connectives (እና, ወይም, ግን, ደግሞ), pronouns (እኔ, እኛ), and demonstratives and quantifiers (ይህ, ሁሉ, አንድ). Every word was passed through the same normalization and tokenizer used for the corpus, duplicates were removed, and each word was checked against the vocabulary. All 86 words were found in the 50,000-word vocabulary, so none had to be discarded.

## Effect on the corpus

The training data is a stream of 68,942,614 token IDs (words plus sentence markers). Removing the stopwords deleted 7,809,464 IDs, which is 11.3% of the stream. Removal also left 7,197 sentences empty; these were dropped, so the number of sentences fell from 4,067,589 to 4,060,392 and the stream shrank to 61,118,756 IDs. The share of removed tokens is lower than one might expect for function words, partly because about 14% of the training tokens (13.94%) are already mapped to `<UNK>`.

As a validation step, we recomputed the n-gram counts of the full model from the same training data and compared the top-10 tables with those produced earlier in Task 1.3. They matched exactly (for example ነው 921,688, ላይ 594,173 and ውስጥ 273,656 for unigrams), which confirms that both runs counted the same data in the same way.

## Effect on the top n-grams

Before removal, the most frequent unigrams were all function words: ነው (921,688), ላይ (594,173), ውስጥ (273,656), ወደ (260,934) and እና (246,281). After removal, the top unigrams are content words: ጊዜ (193,009), ነገር (171,940), ቤት (142,527), መንግስት (131,776) and አቶ (131,651).

The bigrams show the same shift. Phrases that contain a stopword, such as ነገር ግን (35,269), ብቻ ሳይሆን (25,946) and ማለት ነው (25,588), disappear from the top ranks. Noun phrases that were already frequent move up to the top: ዶ ር (28,055), አዲስ አበባ (24,546), በአዲስ አበባ (18,922) and ምክር ቤት (18,825). One new phrase enters the top ten, ጠቅላይ ሚኒስትር (15,789).

The top 4-grams change very little. Most of them were already content phrases, such as የህዝብ ተወካዮች ምክር ቤት (2,257 before, 2,258 after) and የአዲስ አበባ ከተማ አስተዳደር (2,099 both times). The main exception is አ ም ኢሳት ዜና, which disappears because ም is on the stopword list.

The counts of n-grams that contain no stopword stay the same or rise slightly, because deleting a stopword can join words that were separated before. Their probabilities rise more clearly, because the total number of tokens is smaller. For example, the probability of ጊዜ increases from 0.00298 to 0.00338.

## Side effects of removal

Removing a stopword joins the words on either side of it, so some n-grams in the filtered text never occurred in the original. For example, አ ኢሳት ዜና comes from አ ም ኢሳት ዜና after ም is removed. The list includes ም, but in this corpus it often appears as the second half of the abbreviation ዓ.ም. (the bigram አ ም occurs 81,929 times), so removing it also breaks that phrase. More generally, our tokenizer splits abbreviations into single letters (አ ም, ዶ ር, እ ኤ አ), so fragments such as the unigram አ (108,143 occurrences) and the trigram እ ኤ አ appear among the most frequent results. These are artifacts of tokenization, not real words. Finally, the results depend on this particular list; a longer or shorter list would give different counts.

## Word clouds

We generated eight word clouds, one for each n-gram order (1 to 4) before and after stopword removal, using the Noto Sans Ethiopic font so that the Amharic letters display correctly. Each cloud shows the 150 most frequent n-grams, with the size of each entry proportional to its count. In the unigram cloud before removal, the largest entries are function words such as ወደ, ነው, ላይ, እና and ውስጥ; after removal, content words such as ጊዜ, ነገር, ቤት, መንግስት, ሰው and ሰዎች take their place. The bigram clouds show the same change: before removal, ነገር ግን, ማለት ነው, ብቻ ሳይሆን and ብቻ ነው are prominent, and after removal they are gone, while noun phrases such as አዲስ አበባ, ምክር ቤት and ጠቅላይ ሚኒስትር become the largest entries. The trigram clouds change in the same way: phrases such as ከጊዜ ወደ ጊዜ and ነው ማለት ነው disappear. Fragments of broken abbreviations (እ ኤ አ, ዶ ር, and አ ኢሳት after removal) are visible in the clouds both before and after removal.

The 4-gram clouds look almost the same before and after, apart from አ ም ኢሳት ዜና, which disappears, because the most frequent 4-grams were already content phrases. They are dominated by text that is repeated across many web-news articles, such as the ESAT news tag, the disclaimer that an article reflects only the writer's opinion, and the headings of the Golgul web newspaper and its "My Opinion" column. This fits the large share of duplicate sentences in the corpus.
