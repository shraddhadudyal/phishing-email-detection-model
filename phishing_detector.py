"""
Phishing Email Detection Model
Educational demo using scikit-learn TF-IDF and Logistic Regression.
Run: python phishing_detector.py
"""
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "emails.csv"
MODEL_PATH = BASE_DIR / "phishing_model.joblib"

def extract_simple_features(text: str) -> str:
    """Add lightweight URL/count indicators alongside the original email text."""
    text = str(text)
    url_count = len(re.findall(r"https?://|www\.", text, flags=re.IGNORECASE))
    exclamation_count = text.count("!")
    digit_count = sum(char.isdigit() for char in text)
    # These tokens let the text model learn from a few simple content/URL signals.
    indicators = []
    indicators.extend([" URL_PRESENT"] * min(url_count, 3))
    indicators.extend([" EXCLAMATION_MARK"] * min(exclamation_count, 3))
    indicators.extend([" MANY_DIGITS"] * (1 if digit_count >= 6 else 0))
    return text + " " + " ".join(indicators)

def build_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), strip_accents="unicode")),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("emails.csv is missing. Keep it in the same folder as this script.")
    data = pd.read_csv(DATA_PATH).dropna(subset=["text", "label"])
    data["text"] = data["text"].astype(str)
    data["label"] = data["label"].astype(str).str.strip().str.title()
    data = data[data["label"].isin(["Safe", "Phishing"])].drop_duplicates()

    if data["label"].nunique() != 2 or data["label"].value_counts().min() < 2:
        raise ValueError("Dataset must contain at least two examples for each label: Safe and Phishing.")

    X = data["text"].map(extract_simple_features)
    y = data["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = build_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    labels = ["Phishing", "Safe"]
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    print("\n" + "=" * 58)
    print("       PHISHING EMAIL DETECTION MODEL")
    print("=" * 58)
    print(f"Total examples: {len(data)} | Training: {len(X_train)} | Testing: {len(X_test)}")
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, labels=labels, zero_division=0))
    print("Confusion matrix (rows = actual, columns = predicted)")
    print(pd.DataFrame(matrix, index=labels, columns=labels))

    # Save confusion-matrix image for the assignment/report.
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    display.plot(values_format="d")
    plt.title("Phishing Email Detection — Confusion Matrix")
    plt.tight_layout()
    chart_path = BASE_DIR / "confusion_matrix.png"
    plt.savefig(chart_path, dpi=160)
    plt.close()
    print(f"\nConfusion matrix image saved to: {chart_path.name}")

    # Train a final model on the full provided dataset for the interactive demo.
    final_model = build_model()
    final_model.fit(data["text"].map(extract_simple_features), y)
    try:
        import joblib
        joblib.dump(final_model, MODEL_PATH)
        print(f"Trained model saved to: {MODEL_PATH.name}")
    except Exception as exc:
        print(f"Model file could not be saved ({exc}). Prediction demo can still run now.")

    print("\nTry your own sample email below. Type 'quit' to stop.")
    while True:
        try:
            email_text = input("\nPaste email subject/body: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if email_text.lower() in {"quit", "exit", "q"}:
            print("Exiting.")
            break
        if not email_text:
            print("Please enter some email text.")
            continue
        prepared = extract_simple_features(email_text)
        prediction = final_model.predict([prepared])[0]
        probabilities = final_model.predict_proba([prepared])[0]
        confidence = max(probabilities) * 100
        print(f"Prediction: {prediction} (model confidence: {confidence:.1f}%)")
        print("Reminder: this small demo model can be wrong. Do not click suspicious links or share credentials.")

if __name__ == "__main__":
    main()
