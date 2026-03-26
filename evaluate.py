"""
evaluate.py
──────────────────────────────────────────────────────────────────────────────
HOA Task Auto — Final evaluation on the held-out test set.

Run this ONLY after you are done tuning your prompt with train.py.
Reports full classification metrics and saves them to models/eval_report.json.
──────────────────────────────────────────────────────────────────────────────
"""

import json
import time
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm

from dataset import load_split, load_config
from model import HOAClassifier


def evaluate(
    classifier: HOAClassifier,
    test_df: pd.DataFrame,
    report_path: str = "models/eval_report.json",
) -> dict:
    """
    Run classifier on the full test set and compute evaluation metrics.

    Args:
        classifier  : HOAClassifier instance
        test_df     : Test DataFrame with columns [text, label]
        report_path : File path to save the JSON report

    Returns:
        dict with accuracy, per-class F1, and confusion matrix
    """
    Path("models").mkdir(exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  FINAL EVALUATION  |  {len(test_df)} test samples")
    print(f"{'═'*60}\n")
    print("⚠  Using held-out test set — do not re-run after reviewing results.\n")

    predictions, labels = [], []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
        try:
            result = classifier.classify(row["text"])
            pred = result["category"]
        except Exception:
            pred = "General Inquiry"

        predictions.append(pred)
        labels.append(row["label"])
        time.sleep(0.1)

    # ── Metrics ───────────────────────────────────────────────────────────────
    overall_acc = accuracy_score(labels, predictions)
    report_str = classification_report(labels, predictions, digits=4)
    report_dict = classification_report(labels, predictions, digits=4, output_dict=True)
    cm = confusion_matrix(labels, predictions,
                          labels=list(report_dict.keys())[:-3])

    print("\n" + report_str)
    print(f"\n{'─'*60}")
    print(f"  Overall Accuracy : {overall_acc:.4f}  ({overall_acc:.1%})")
    print(f"{'─'*60}\n")

    # ── Confusion matrix (text) ───────────────────────────────────────────────
    categories = sorted(set(labels))
    cm_df = pd.DataFrame(cm, index=categories, columns=categories)
    print("Confusion Matrix:")
    print(cm_df.to_string())
    print()

    # ── Save report ───────────────────────────────────────────────────────────
    output = {
        "overall_accuracy": round(float(overall_acc), 4),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "categories": categories,
        "n_test_samples": len(test_df),
    }
    with open(report_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"✔  Report saved to {report_path}")

    return output


# ── CLI entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = load_config()

    print("Loading test set …")
    test_df = load_split("test", data_dir="data")
    print(f"Test samples: {len(test_df)}")

    print("Initialising HOA Classifier (Claude Sonnet 4.6) …")
    classifier = HOAClassifier()

    results = evaluate(
        classifier=classifier,
        test_df=test_df,
        report_path=cfg["eval"]["report_path"],
    )

    target = cfg["eval"]["target_accuracy"]
    acc = results["overall_accuracy"]
    status = "✅ TARGET MET" if acc >= target else f"⚠  Below target ({target:.0%})"
    print(f"\nFinal Test Accuracy: {acc:.1%}  {status}")
