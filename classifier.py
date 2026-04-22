import csv
import re
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Regex patterns for preprocessing
TAG_RE = re.compile(r"<[^>]+>")
TOKEN_RE = re.compile(r"\b[a-z]+\b")
SPACE_RE = re.compile(r"\s+")

# Tools for the text normalization
STEMMER = PorterStemmer()
STOP_WORDS = set(stopwords.words("english"))


def preprocess(text: str) -> str:
    # lowerse and removing HTML tags
    text = TAG_RE.sub(" ", text.lower())

    # Extracting alphabetic tokens
    tokens = TOKEN_RE.findall(text)

    # Removing stopwords
    tokens = [t for t in tokens if t not in STOP_WORDS]

    # Applying stemming
    tokens = [STEMMER.stem(t) for t in tokens]

    # Rejoining cleaned tokens
    return SPACE_RE.sub(" ", " ".join(tokens)).strip()


def load_lexicon(vocab_path: Path, er_path: Path, threshold=0.15):
    lexicon = {}

    # Reading vocabulary and expected ratings together - does not need preprocessing 
    with vocab_path.open("r", encoding="utf-8") as v_file, er_path.open("r", encoding="utf-8") as er_file:
        for word, rating_str in zip(v_file, er_file):
            word = word.strip()
            rating = float(rating_str.strip())

            # Only keeping the words with strong sentiment
            if abs(rating) >= threshold:
                lexicon[word] = rating

    return lexicon


def predict_review(text: str, lexicon: dict) -> int:
    # Spliting the review into words
    words = text.split()

    # Summing sentiment scores of all words
    total_score = sum(lexicon.get(w, 0.0) for w in words)

    # Applying threshold to classify
    if total_score < -0.5:
        return 0  # Negative
    return 1      # Positive


def load_raw_test_reviews(dataset_dir: Path):
    raw_reviews = []

    for label_name, label in [("pos", 1), ("neg", 0)]:
        folder = dataset_dir / "test" / label_name
        for file_path in sorted(folder.glob("*.txt")):
            raw = file_path.read_text(encoding="utf-8", errors="ignore")
            raw_reviews.append((raw, label))

    return raw_reviews


def clean_for_display(text: str, limit: int) -> str:
    return SPACE_RE.sub(" ", text).strip()[:limit]


def main():
    dataset_dir = Path("aclImdb")
    vocab_file = dataset_dir / "imdb.vocab"
    er_file = dataset_dir / "imdbEr.txt"
    test_csv = Path("processed") / "test.csv"

    print("Loading weighted sentiment lexicon...")
    lexicon = load_lexicon(vocab_file, er_file)

    print(f"Loaded {len(lexicon)} words with expected ratings.")
    print("Evaluating test set...")

    correct_predictions = 0
    total_reviews = 0

    # Counters for evaluation metrics
    tp = tn = fp = fn = 0

    # Store examples
    errors = []
    correct_pos = []
    correct_neg = []

    raw_reviews = load_raw_test_reviews(dataset_dir)

    with test_csv.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # Skipping header

        for index, row in enumerate(reader):
            if not row or len(row) != 2:
                continue

            text, actual_label = row[0], int(row[1])
            raw_text, raw_label = raw_reviews[index]

            if actual_label != raw_label:
                continue

            raw_display = clean_for_display(raw_text, 220)
            processed_display = clean_for_display(text, 160)

            # Predicting sentiment
            predicted_label = predict_review(text, lexicon)

            # Tracking accuracy
            if predicted_label == actual_label:
                correct_predictions += 1

                # Saving a few correct examples
                if actual_label == 1 and len(correct_pos) < 1:
                    correct_pos.append((raw_display, processed_display))
                elif actual_label == 0 and len(correct_neg) < 1:
                    correct_neg.append((raw_display, processed_display))

            #   tracking precision/recall counts
            if predicted_label == 1 and actual_label == 1:
                tp += 1
            elif predicted_label == 0 and actual_label == 0:
                tn += 1
            elif predicted_label == 1 and actual_label == 0:
                fp += 1
            elif predicted_label == 0 and actual_label == 1:
                fn += 1

            # Saving misclassified examples
            if predicted_label != actual_label and len(errors) < 1:
                errors.append((raw_display, processed_display, actual_label, predicted_label))

            total_reviews += 1

    if total_reviews > 0: # calculating metrics
        accuracy = (correct_predictions / total_reviews) * 100
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        print("\nResults:")
        print(f"Total Reviews Evaluated: {total_reviews}")
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")

        # displaying correct examples
        print("\nCorrect Positive Examples:")
        for raw_ex, processed_ex in correct_pos:
            print("Original Review:")
            print(raw_ex, "...")
            print("Processed Review:")
            print(processed_ex, "...")

        print("\nCorrect Negative Examples:")
        for raw_ex, processed_ex in correct_neg:
            print("Original Review:")
            print(raw_ex, "...")
            print("Processed Review:")
            print(processed_ex, "...")

        # Showing errors
        print("\nSample Misclassified Reviews:")
        for raw_text, processed_text, actual, predicted in errors:
            print(f"\nActual: {'Positive' if actual==1 else 'Negative'} | Predicted: {'Positive' if predicted==1 else 'Negative'}")
            print("Original Review:")
            print(raw_text, "...")
            print("Processed Review:")
            print(processed_text, "...")

    else:
        print("No reviews found.")
        return

    # Phase 2: Interactive (user can test their own reviews)
    print("\nInteractive Review Tester (type 'exit' to quit)")

    while True:
        user_input = input("\nEnter a review: ").strip()
        if user_input.lower() == "exit":
            break
            # Preprocessed because unlike previous reviews - the user input is not cleaned (raw)
        processed = preprocess(user_input)
        prediction = predict_review(processed, lexicon)

        print("Prediction:", "Positive" if prediction == 1 else "Negative")


if __name__ == "__main__":
    main()