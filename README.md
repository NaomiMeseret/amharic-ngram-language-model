# Amharic N-gram Language Modeling Project

This group project builds and analyzes N-gram language models for Amharic using the GPAC text corpus. The work completed so far includes full-corpus preprocessing, four sparse N-gram orders, model probability methods, and saved analysis reports. The repository also provides the shared preprocessing rules used by model inference.

## Project workflow

### 1. Corpus preprocessing

The preprocessing pipeline normalizes Amharic spelling, tokenizes words and numbers, splits text into complete sentences, assigns sentences deterministically to train, development, and test splits, and builds a 50,000-entry vocabulary from training data. Processed arrays are flattened `int32` vocabulary IDs with `<s>` and `</s>` sentence markers. The original corpus and processed arrays are not tracked in Git; use the prepared data files supplied to the group.

### 2. N-gram model building

The model code builds unigram, bigram, trigram, and four-gram counts without crossing sentence boundaries. It supports maximum-likelihood probabilities, add-k smoothing, linear interpolation, absolute-discount backoff, log probabilities, and next-word suggestions. The disk-backed trainer uses sorted intermediate runs and checkpoints to handle the full training split with bounded memory.

The saved full-run report records 4,067,589 training sentences and 64,875,025 prediction events per order. Unique N-gram counts are 49,998 unigrams, 10,265,086 bigrams, 27,987,839 trigrams, and 38,108,498 four-grams. See `report/training_report.json` for the recorded training details.

### 3. Analysis completed

The current analysis outputs include:

- Top-10 N-grams for each order, with counts and empirical joint probabilities.
- Bigram conditional probability examples, with unsmoothed and add-1 values.
- Unsmoothed and add-1 probabilities for the assignment sentence `ኢትዮጵያ በአፍሪካ ቀንድ ትገኛለች።`.
- Development-set tuning reports for add-k values and interpolation weights.
- Probability normalization checks.

These compact CSV and JSON files are in `results/analysis/`.

### 4. Remaining group integration

The repository does not currently contain completed reports for stopword experiments, official perplexity or test-set evaluation, random sentence generation, or Top-1/Top-5 predictive-text evaluation. Coordinate these items with the members responsible for them. The test split must remain reserved for final evaluation.

## Repository layout

- `src/preprocess.py`, `src/preprocess_lib.py` — preprocessing pipeline and shared text rules.
- `src/ngram_model.py` — sparse model counts, probabilities, smoothing, interpolation, backoff, and prediction.
- `src/ngram_disk.py` — disk-backed counting and resumable checkpoints.
- `src/train_ngram_slice.py` — complete-sentence prefix selection and small-slice validation.
- `tests/` — unit tests for model counts, sentence boundaries, probability methods, and disk-backed recovery.
- `notebooks/preprocessing.ipynb` — preprocessing workflow.
- `notebooks/amharic-n-gram-model-building-and-training.ipynb` — Kaggle full-training and model-analysis workflow.
- `notebooks/02_ngram_models.ipynb` — earlier sample-model experiments.
- `results/analysis/` — compact assignment outputs.
- `report/` — training and artifact reports.

## Data and setup

The code expects a matching processed training array and vocabulary, normally `data/processed/train.npy` and `data/vocab.txt`. Keep these files from the same preprocessing run so their token IDs agree. Large arrays and trained model tables are excluded from Git; use Kaggle outputs or another shared artifact location to distribute them.

Use Python 3.10 or later with NumPy. From the repository root:

```bash
python3 -m pip install -r requirements.txt
```

Run the unit tests with:

```bash
python3 -m unittest discover -s tests -v
```

To validate a small prefix of the training array, run:

```bash
python3 -m src.train_ngram_slice \
  --max-tokens 2000000 \
  --output-dir results/ngram_slices/train_2m
```

The slice validator uses `data/processed/train.npy` and `data/vocab.txt` by default, and refuses to overwrite an existing output directory.

## Running full training on Kaggle

Attach the processed-data dataset and the project code to the Kaggle notebook. Full training uses the training split only and writes model and report artifacts under `/kaggle/working/ngram_results/`. Kaggle working files are session outputs; download or save needed artifacts before the session ends. The full-training notebook cell can launch a lengthy run, so do not use **Run All** unless you intend to run training.

## Probability conventions

Each sentence is counted separately. `<s>` supplies beginning context but is not a prediction target; `</s>` is predicted once per sentence. The prediction vocabulary includes `<UNK>` and `</s>`, and excludes `<PAD>` and `<s>`. Add-k smoothing uses:

`P(w | h) = (count(h, w) + k) / (count(h) + k × prediction_vocabulary_size)`

The top-10 probability files show empirical joint N-gram probabilities; the bigram conditional report separately gives `P(next word | previous word)`.
