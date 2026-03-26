"""
dataset.py
──────────────────────────────────────────────────────────────────────────────
HOA Task Auto — Dataset loading, generation, and preprocessing.

Responsibilities:
  • Generate a synthetic labelled dataset of HOA resident requests
  • Load / save CSV splits (train / val / test)
  • Provide a clean DataLoader-style iterator for evaluation
──────────────────────────────────────────────────────────────────────────────
"""

import random
import pandas as pd
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split


# ── Load config ───────────────────────────────────────────────────────────────
def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Synthetic request templates per category ─────────────────────────────────
TEMPLATES = {
    "Maintenance Request": [
        "The street light outside unit {n} has been out for a week.",
        "There is a water leak in the common hallway near building {n}.",
        "The elevator in block {n} is making a loud noise.",
        "Broken gate at entrance {n} needs urgent repair.",
        "Pool pump at amenity area {n} stopped working.",
        "The parking lot lights in section {n} are not working.",
        "Fence along property line {n} has fallen over.",
        "Air conditioning in common area {n} is not cooling properly.",
    ],
    "Dues & Payment": [
        "I would like to set up automatic payment for my monthly dues.",
        "I haven't received my invoice for this quarter yet.",
        "Can I get a receipt for my last payment of unit {n}?",
        "I need to update my payment method on file.",
        "Is there a payment plan available for overdue balances?",
        "I believe I was double charged for unit {n} this month.",
        "When is the deadline for annual assessment payment?",
        "Please confirm my account balance for unit {n}.",
    ],
    "Rule Violation Report": [
        "My neighbor at unit {n} is parking in a handicap spot without a permit.",
        "There is unauthorized construction happening at unit {n}.",
        "A resident at building {n} is storing items in the common hallway.",
        "Loud music from unit {n} after quiet hours.",
        "Unapproved pet over weight limit seen at unit {n}.",
        "Resident at unit {n} has installed an unauthorized satellite dish.",
        "Commercial vehicle parked in residential spot near unit {n}.",
        "Smoking in non-designated area near building {n}.",
    ],
    "Amenity Booking": [
        "I'd like to reserve the clubhouse for a birthday party on Saturday.",
        "Can I book the BBQ area near pool {n} for this weekend?",
        "I need to reserve the community room for a meeting next Tuesday.",
        "How do I book a guest parking spot for the weekend?",
        "Is the tennis court near area {n} available this Sunday morning?",
        "I want to schedule the pool area for a private event.",
        "Please reserve the gym for a group fitness session on Friday.",
        "Can I book the rooftop lounge for a small gathering?",
    ],
    "General Inquiry": [
        "What are the office hours for the HOA management team?",
        "How do I obtain a copy of the community rules and regulations?",
        "Where can I find information about the upcoming board meeting?",
        "What is the process for getting a parking permit for unit {n}?",
        "Who do I contact for after-hours emergencies?",
        "Is there a community newsletter I can subscribe to?",
        "What recycling guidelines does our community follow?",
        "How do I submit a suggestion to the HOA board?",
    ],
    "Complaint / Dispute": [
        "I am disputing the fine I received for unit {n} last month.",
        "My neighbor's tree is dropping leaves into my patio at unit {n}.",
        "I feel the recent rule change was not communicated properly.",
        "The contractor hired by HOA damaged my property at unit {n}.",
        "I was unfairly charged a late fee for my payment.",
        "The board meeting minutes do not reflect what was discussed.",
        "I disagree with the decision made about unit {n} modification.",
        "I want to appeal the architectural review board decision.",
    ],
}


def generate_dataset(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic labelled HOA request dataset."""
    random.seed(seed)
    records = []
    categories = list(TEMPLATES.keys())
    # Distribute samples roughly per class weights: 28/22/18/15/10/7
    weights = [28, 22, 18, 15, 10, 7]
    total = sum(weights)
    counts = [round(n_samples * w / total) for w in weights]
    # Adjust rounding to hit exactly n_samples
    counts[-1] += n_samples - sum(counts)

    for category, count in zip(categories, counts):
        tmpl_list = TEMPLATES[category]
        for _ in range(count):
            tmpl = random.choice(tmpl_list)
            unit = random.randint(100, 999)
            text = tmpl.replace("{n}", str(unit))
            # Light augmentation: vary sentence casing / punctuation
            if random.random() < 0.2:
                text = text.lower()
            elif random.random() < 0.1:
                text = text.upper()
            records.append({"text": text, "label": category})

    df = pd.DataFrame(records).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def split_and_save(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    out_dir: str = "data",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train/val/test and save to CSV."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    test_ratio = 1.0 - train_ratio - val_ratio

    train, temp = train_test_split(
        df, test_size=(1 - train_ratio), random_state=seed, stratify=df["label"]
    )
    val, test = train_test_split(
        temp,
        test_size=(test_ratio / (val_ratio + test_ratio)),
        random_state=seed,
        stratify=temp["label"],
    )

    train.to_csv(f"{out_dir}/train.csv", index=False)
    val.to_csv(f"{out_dir}/val.csv", index=False)
    test.to_csv(f"{out_dir}/test.csv", index=False)

    print(f"✔  Train : {len(train):>5} samples")
    print(f"✔  Val   : {len(val):>5} samples")
    print(f"✔  Test  : {len(test):>5} samples")
    return train, val, test


def load_split(split: str = "test", data_dir: str = "data") -> pd.DataFrame:
    """Load a pre-saved CSV split.  split ∈ {'train','val','test'}"""
    path = Path(data_dir) / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python dataset.py` first to generate data."
        )
    return pd.read_csv(path)


# ── CLI entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = load_config()
    print("Generating HOA request dataset …")
    df = generate_dataset(n_samples=2000, seed=cfg["data"]["random_seed"])
    print(f"Total samples  : {len(df)}")
    print("Label distribution:")
    print(df["label"].value_counts().to_string())
    print()
    split_and_save(
        df,
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
        seed=cfg["data"]["random_seed"],
        out_dir="data",
    )
    print("\nDataset saved to data/train.csv, data/val.csv, data/test.csv")
