"""
train.py
──────────────────────────────────────────────────────────────────────────────
HOA Task Auto — Training / prompt-engineering loop.

Because HOA Task Auto uses Claude Sonnet 4.6 (a pre-trained LLM) via API,
"training" means iterative prompt engineering and few-shot optimisation rather
than weight updates.  This script:

  1. Loads the validation split
  2. Runs the current prompt against the val set
  3. Identifies misclassified samples per category
  4. Reports per-class accuracy and overall accuracy
  5. Saves the best-performing prompt configuration to models/best_prompt.json

Run this script to tune your prompt before final evaluation on the test set.
──────────────────────────────────────────────────────────────────────────────
"""

import json
import time
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from dataset import load_split, load_config
from model import HOAClassifier


def run_prompt_tuning(
    classifier: HOAClassifier,
    val_df: pd.DataFrame,
    max_samples: int = 100,
    save_path: str = "models/best_prompt.json",
) -> dict:
    """
    Run classifier on a subset of the val set, report per-class accuracy,
    and save results.

    Args:
        classifier   : HOAClassifier instance
        val_df       : Validation DataFrame with columns [text, label]
        max_samples  : Cap samples to avoid excessive API cost during tuning
        save_path    : Where to save the results JSON

    Returns:
        results dict with per-class and overall accuracy
    """
    Path("models").mkdir(exist_ok=True)

    # Sample evenly across classes
    sample_df = (
        val_df.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), max_samples // 6), random_state=42))
        .reset_index(drop=True)
    )

    print(f"\n{'─'*60}")
    print(f"  Prompt Tuning Run  |  {len(sample_df)} val samples")
    print(f"{'─'*60}\n")

    predictions, labels, errors = [], [], []

    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Validating"):
        try:
            result = classifier.classify(row["text"])
            pred = result["category"]
        except Exception as e:
            pred = "General Inquiry"  # safe fallback
            errors.append({"text": row["text"], "error": str(e)})

        predictions.append(pred)
        labels.append(row["label"])
        time.sleep(0.1)  # respect rate limits

    # ── Per-class accuracy ────────────────────────────────────────────────────
    results_df = pd.DataFrame({"text": sample_df["text"].values,
                                "true": labels, "pred": predictions})
    categories = sorted(results_df["true"].unique())
    per_class = {}
    print(f"{'Category':<30} {'Correct':>7} {'Total':>7} {'Accuracy':>9}")
    print("─" * 58)
    for cat in categories:
        subset = results_df[results_df["true"] == cat]
        correct = (subset["pred"] == subset["true"]).sum()
        acc = correct / len(subset) if len(subset) > 0 else 0.0
        per_class[cat] = {"correct": int(correct), "total": len(subset), "accuracy": round(acc, 4)}
        print(f"{cat:<30} {correct:>7} {len(subset):>7} {acc:>9.1%}")

    overall_acc = (results_df["pred"] == results_df["true"]).mean()
    print("─" * 58)
    print(f"{'OVERALL':<30} {(results_df['pred']==results_df['true']).sum():>7} "
          f"{len(results_df):>7} {overall_acc:>9.1%}\n")

    # ── Show misclassified samples ────────────────────────────────────────────
    misclassified = results_df[results_df["pred"] != results_df["true"]]
    if not misclassified.empty:
        print(f"\n⚠  Misclassified samples ({len(misclassified)}):")
        for _, row in misclassified.head(10).iterrows():
            print(f"  True: {row['true']:<28}  Pred: {row['pred']}")
            print(f"  Text: {row['text'][:80]}")
            print()

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        "overall_accuracy": round(float(overall_acc), 4),
        "per_class_accuracy": per_class,
        "n_samples": len(sample_df),
        "n_errors": len(errors),
        "errors": errors[:5],  # save first 5 only
    }
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"✔  Results saved to {save_path}")

    return output


# ── CLI entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = load_config()
    print("Loading validation set …")
    val_df = load_split("val", data_dir="data")
    print(f"Val samples: {len(val_df)}")

    print("Initialising HOA Classifier (Claude Sonnet 4.6) …")
    classifier = HOAClassifier()

    results = run_prompt_tuning(
        classifier=classifier,
        val_df=val_df,
        max_samples=60,           # ~10 per class — fast and cost-effective
        save_path=cfg["eval"]["report_path"].replace("eval_report", "val_report"),
    )

    target = cfg["eval"]["target_accuracy"]
    acc = results["overall_accuracy"]
    status = "✅ TARGET MET" if acc >= target else f"⚠  Below target ({target:.0%})"
    print(f"\nVal Accuracy: {acc:.1%}  {status}")
    print("\nTo improve: edit FEW_SHOT_EXAMPLES or SYSTEM_PROMPT in model.py, then re-run train.py")
