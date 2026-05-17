# IMDb Movie Review Sorter

A lightweight Python NLP pipeline that classifies IMDb movie reviews as **positive** or **negative** using a lexicon-based sentiment scoring approach — no machine learning training required.

## Overview

The project has two scripts that work in sequence:

1. `reviewer.py` — preprocesses the raw IMDb dataset and saves cleaned reviews to CSV files
2. `classifier.py` — loads the processed reviews, scores them against a weighted sentiment lexicon, evaluates accuracy/precision/recall, and provides an interactive mode for testing custom reviews

## How It Works

### Text Preprocessing (both scripts)

Each review goes through the same normalization pipeline:

- Strip HTML tags
- Lowercase all text
- Extract only alphabetic tokens
- Remove English stopwords (via NLTK)
- Apply Porter stemming

### Sentiment Scoring (`classifier.py`)

The classifier uses the IMDb vocabulary file (`imdb.vocab`) paired with expected ratings (`imdbEr.txt`) to build a sentiment lexicon. Words with an absolute rating below a configurable threshold (default `0.15`) are discarded. At prediction time, the scores of all words in a review are summed — a total below `-0.5` is classified as negative, anything else as positive.

### Evaluation Output

After running on the test set, the classifier prints:

- Total reviews evaluated
- Number of correct predictions
- Accuracy, Precision, and Recall
- Sample correct positive and negative predictions
- Sample misclassified reviews (with both raw and processed text shown)

## Project Structure

```
IMDb-Movie/
├── reviewer.py      # Preprocessing: reads aclImdb/, writes processed/train.csv and test.csv
└── classifier.py    # Classification: evaluates test set + interactive review tester
```

## Requirements

- Python 3.8+
- [NLTK](https://www.nltk.org/) with the `stopwords` corpus

Install dependencies:

```bash
pip install nltk
python -c "import nltk; nltk.download('stopwords')"
```

## Dataset Setup

This project uses the [Large Movie Review Dataset (aclImdb)](https://ai.stanford.edu/~amaas/data/sentiment/) by Andrew Maas et al.

Download and extract it so your directory looks like:

```
aclImdb/
├── imdb.vocab
├── imdbEr.txt
├── train/
│   ├── pos/   ← positive review .txt files
│   └── neg/   ← negative review .txt files
└── test/
    ├── pos/
    └── neg/
```

## Usage

**Step 1 — Preprocess the dataset:**

```bash
python reviewer.py
```

This reads all `.txt` files from `aclImdb/train/` and `aclImdb/test/`, preprocesses them, and writes two CSVs:

```
processed/
├── train.csv
└── test.csv
```

**Step 2 — Run the classifier:**

```bash
python classifier.py
```

This evaluates the test set and then enters an interactive loop where you can type any review and get an instant positive/negative prediction:

```
Enter a review: The cinematography was breathtaking and the acting superb.
Prediction: Positive

Enter a review: exit
```

## Configuration

In `classifier.py`, the `load_lexicon` function accepts a `threshold` parameter (default `0.15`) that controls how strongly opinionated a word must be to be included in the lexicon. Raising it makes the lexicon more selective; lowering it includes more neutral words.

## Dataset Citation

> Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). *Learning Word Vectors for Sentiment Analysis.* ACL 2011.
