<div align="center">

# 🎬 IMDb Movie Review Classifier
### Lexicon-Based Sentiment Analysis — No Model Training Required

A Python NLP pipeline that classifies IMDb movie reviews as **positive or negative** using a **weighted sentiment lexicon** built directly from the IMDb vocabulary files — no machine learning training loop, no neural networks, just scored word matching.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-Stopwords_%26_Stemming-009900?style=for-the-badge&logo=python&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Sentiment_Analysis-FF6600?style=for-the-badge&logo=openai&logoColor=white)
![IMDb](https://img.shields.io/badge/Dataset-IMDb_aclImdb-F5C518?style=for-the-badge&logo=imdb&logoColor=black)

</div>

---

## How It Works

**Step 1 — Preprocessing (`reviewer.py`)**
1. Reads all `.txt` review files from `aclImdb/train/` and `aclImdb/test/`
2. Each review is cleaned: HTML tags stripped, lowercased, punctuation removed, stopwords filtered, Porter stemming applied
3. Cleaned reviews are saved to `processed/train.csv` and `processed/test.csv`

**Step 2 — Classification (`classifier.py`)**
1. Loads `imdb.vocab` paired with `imdbEr.txt` to build a **weighted sentiment lexicon** — words with an absolute rating below `0.15` are discarded
2. Each review's words are scored by summing their lexicon values
3. A total below `-0.5` → **Negative**, anything else → **Positive**
4. Accuracy, precision, and recall are printed along with sample correct and misclassified reviews
5. An **interactive mode** lets you type any review and get an instant prediction

**Accuracy achieved:** ~70%+ on the IMDb test set using zero model training

---

## Setup

**Requirements:** Python 3.11+ · NLTK

**1. Clone & install**
```bash
git clone https://github.com/aminabk99/IMDb-Movie
cd IMDb-Movie
pip install nltk
python -c "import nltk; nltk.download('stopwords')"
```

**2. Download the dataset**

Download the [Large Movie Review Dataset (aclImdb)](https://ai.stanford.edu/~amaas/data/sentiment/) and extract it so your directory looks like:
aclImdb/
├── imdb.vocab
├── imdbEr.txt
├── train/
│   ├── pos/
│   └── neg/
└── test/
├── pos/
└── neg/

**3. Preprocess**
```bash
python reviewer.py
```

**4. Classify + interact**
```bash
python classifier.py
```

---

```
## Project Structure
IMDb-Movie/
├── reviewer.py       # Preprocessing pipeline — cleans raw reviews, writes CSVs
├── classifier.py     # Lexicon scoring, evaluation metrics, interactive tester
├── processed/        # Auto-generated after running reviewer.py
│   ├── train.csv
│   └── test.csv
└── aclImdb/          # Dataset (not included — download separately)

```
---

## Hardest Part
**Calibrating the lexicon threshold** — setting it too low included too many neutral words that drowned out signal, too high left too few words to score anything. The `0.15` cutoff and `-0.5` classification boundary were tuned by hand against the test set.

## Most Interesting
**Getting solid accuracy without training a single model** — the entire classification logic is a sum of pre-rated word scores. It makes the decision boundary completely transparent: you can read exactly why a review was classified the way it was, which no neural net can give you.

---

## Dataset Citation

> Andrew L. Maas et al. (2011). *Learning Word Vectors for Sentiment Analysis.* ACL 2011.

---

<div align="center">
  <sub>Built by <a href="https://github.com/aminabk99">Amina Bilal</a> · <a href="https://linkedin.com/in/amina-bilal-926340382">LinkedIn</a></sub>
</div>
