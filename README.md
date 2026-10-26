# Amharic N-Gram Language Modeling Project

A group project that preprocesses Amharic text and builds unigram, bigram, trigram, and four-gram language models from the GPAC corpus. The project includes disk-backed training, probability analysis, intrinsic and extrinsic evaluation, stopword experiments, sentence generation, and a React next-word suggestion interface.

## Team members

| Member | Name | Student ID |
|---|---|---|
| Member 1 | Feven Tewelde | UGR/0805/15 |
| Member 2 | Mahlet Adane | UGR/7853/15 |
| Member 3 | Meklit Workineh | UGR/3445/15 |
| Member 4 | Naomi Meseret | UGR/5048/15 |
| Member 5 | Rahel Yabebal | UGR/6200/15 |

## Dataset and preprocessing

The preprocessing pipeline is implemented in `src/preprocess.py`; shared normalization and tokenization rules are in `src/preprocess_lib.py`. It applies Unicode NFC normalization and Amharic spelling mappings, tokenizes words and numbers, splits the text into sentences, assigns normalized sentences deterministically to train, development, and test splits, and builds a 50,000-entry vocabulary from training data only. Sentences longer than 100 tokens after tokenization are excluded. The saved arrays are flattened `int32` vocabulary IDs with `<s>` and `</s>` sentence-boundary markers.

Recorded full-corpus split sizes:

| Split | Sentences | Token IDs |
|---|---:|---:|
| Train | 4,067,589 | 68,942,614 |
| Development | 508,114 | 8,605,850 |
| Test | 504,690 | 8,579,227 |

The original corpus and processed `.npy` arrays are not committed to Git. Use the group’s shared storage to obtain the arrays and their matching vocabulary; token IDs are only valid with the vocabulary from the same preprocessing run.

## Model and full training

`src/ngram_model.py` implements sparse counts and probability queries for n-gram orders 1–4. It supports maximum-likelihood estimation (MLE), add-k smoothing, linear interpolation, absolute-discount backoff, `prob()`, `logprob()`, sentence probabilities, and the shared `predict_next()` interface. Counts are sentence-local: `<s>` supplies context but is not predicted, while `</s>` is a prediction target.

`src/ngram_disk.py` provides memory-mapped, disk-backed counting with sorted intermediate runs, merge-by-key, and checkpoint-based resumption. The full model was fitted using the training split only.

| Order | Occurrences | Unique n-grams |
|---|---:|---:|
| Unigram | 64,875,025 | 49,998 |
| Bigram | 64,875,025 | 10,265,086 |
| Trigram | 64,875,025 | 27,987,839 |
| Four-gram | 64,875,025 | 38,108,498 |

The saved sparse model tables occupy about 1.70 GiB. The recorded Kaggle training run took about 19 minutes and used about 1.82 GiB peak resident memory. These are measurements from that run; resource use varies by environment. See `report/training_report.json` for the training configuration and metadata. Model tables, checkpoints, and large data files are excluded from Git and should be shared as artifacts.

### Probability conventions

The prediction vocabulary contains 49,998 outcomes: all vocabulary IDs except `<PAD>` and `<s>`, including `<UNK>` and `</s>`. Add-k smoothing is:

```text
P(w | h) = (count(h, w) + k) / (count(h) + k × prediction_vocabulary_size)
```

## Results

### N-gram analysis and stopwords

`results/analysis/` contains top-10 n-grams for each order, all-token and word-only tables, empirical n-gram probabilities, bigram conditional probabilities, the assignment sentence probabilities, development tuning grids, and probability normalization checks. `results/stopwords/` contains before/after n-gram frequency tables and word clouds. Methods and observations are documented in `report/stopwords_and_wordclouds.md`.

### Intrinsic evaluation and generation

Perplexity settings were selected on the development split. The table reports tuned add-k results on the whole test set and a separate matched 500-sentence comparison for backoff:

| Order | Tuned add-k, whole test set | Backoff, 500-sentence sample |
|---|---:|---:|
| Unigram | 2,428 | 2,378 |
| Bigram | **538** | 458 |
| Trigram | 1,910 | **409** |
| Four-gram | 6,483 | 410 |

The **best recorded intrinsic result** was perplexity **334** from the four-order linear interpolation model on the same 500-sentence comparison sample. Its development-tuned weights were `[0.2, 0.4, 0.1, 0.3]` for unigram through four-gram. The score is for that sample, not the whole test set. The intrinsic evaluation and generated sentence examples are documented in `report/intrinsic_evaluation and generation.md`.

### Extrinsic evaluation

The sampled next-word evaluation used 2,000 test sentences and exact-order n-gram count ranking. The trigram achieved the best overall Top-1 and Top-5 scores in this comparison.

| Order | Top-1 | Top-5 | Context coverage |
|---|---:|---:|---:|
| Unigram | 1.75% | 5.46% | 100.0% |
| Bigram | 10.80% | 22.08% | 100.0% |
| Trigram | **13.17%** | **22.50%** | 85.5% |
| Four-gram | 11.42% | 16.30% | 54.2% |

The four-gram achieved 21.10% Top-1 accuracy on positions where its context appeared in training, but had lower overall coverage. Real-versus-shuffled sentence discrimination on 998 usable test sentences scored 98.7%–98.9% for orders 2–4; the unigram scored 50.0%, with all examples tied. See `report/extrinsic_evaluation.md` for methods and limitations.

### Sentence generation

The project generates sample sentences with each n-gram order. As n increases, local word sequences become more fluent, but the models encounter more unseen contexts and reuse more phrases from training. Examples and generation statistics are in `report/intrinsic_evaluation and generation.md`.

## React interface

The React/Vite client is in `frontend/`. The FastAPI backend must be running before the client can request predictions. Start the backend from the repository root in **Terminal 1**:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

The backend loads the model from `results/model/`. Make sure that directory contains the trained model files, including `metadata.json` and the order tables. Keep Terminal 1 running.

In **Terminal 2**, start the frontend from the repository root:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite, usually `http://localhost:5173`. The client sends next-word requests to `http://127.0.0.1:8000/predict` and text-generation requests to `http://127.0.0.1:8000/generate`.

## Setup and tests

Use Python 3.10 or later. On macOS, install Python dependencies inside a virtual environment rather than into the system Python. From the repository root, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The test suite covers n-gram counts, sentence boundaries, probability methods, prediction behavior, disk-backed merging, and checkpoint recovery.

## Loading the saved model

The FastAPI app loads the trained model from `results/model/`. If you are querying it directly from Python, use the same path and memory-map the sparse tables:

```python
from src.ngram_model import NGramModel

model = NGramModel.load("results/model", mmap_mode="r")
suggestions = model.predict_next("ኢትዮጵያ በአፍሪካ", k=5)

for word, probability in suggestions:
    print(word, probability)
```

The default method is backoff. Returned scores are conditional probabilities and are not renormalized over only the displayed five suggestions.
## Repository layout

```text
src/                 Preprocessing, n-gram model, disk trainer, and evaluation code
tests/               Unit tests for models, sentence boundaries, and disk training
notebooks/           Preprocessing, training, stopword, and evaluation workflows
frontend/            React and Vite next-word suggestion interface
data/stopwords_am.txt Amharic stopword list
results/analysis/    N-gram counts, probabilities, and tuning outputs
results/stopwords/   Before/after stopword results and word clouds
report/              Training, preprocessing, and evaluation reports
```

The full training notebook is designed for Kaggle. It uses `train.npy` and the matching vocabulary for fitting; development and test data remain separate. Avoid running all notebook cells unless you intend to execute each workflow, including any lengthy training cells.
