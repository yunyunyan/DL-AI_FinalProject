"""
inference.py
──────────────────────────────────────────────────────────────────────────────
HOA Task Auto — Single-request inference pipeline.

Usage:
    python inference.py "The pool pump stopped working near building 3."
    python inference.py  (interactive mode — prompts for input)
──────────────────────────────────────────────────────────────────────────────
"""

import sys
import json
from model import HOAClassifier

# ── Routing table: category → department + action ────────────────────────────
ROUTING = {
    "Maintenance Request": {
        "department": "Maintenance Team",
        "action":     "Create work order and schedule inspection within 48 hours.",
        "priority":   "High",
    },
    "Dues & Payment": {
        "department": "Finance Office",
        "action":     "Review account and respond within 1 business day.",
        "priority":   "Medium",
    },
    "Rule Violation Report": {
        "department": "Compliance Officer",
        "action":     "Log violation report and investigate within 5 business days.",
        "priority":   "High",
    },
    "Amenity Booking": {
        "department": "Community Manager",
        "action":     "Check availability and confirm booking within 24 hours.",
        "priority":   "Low",
    },
    "General Inquiry": {
        "department": "HOA Front Desk",
        "action":     "Provide information or redirect to appropriate department.",
        "priority":   "Low",
    },
    "Complaint / Dispute": {
        "department": "Board Relations",
        "action":     "Log complaint and schedule review at next board meeting.",
        "priority":   "Medium",
    },
}


def process_request(text: str, classifier: HOAClassifier) -> dict:
    """
    Full inference pipeline for a single HOA request.

    Args:
        text       : Resident's free-text message
        classifier : HOAClassifier instance

    Returns:
        dict with category, confidence, reasoning, department, action, priority
    """
    result = classifier.classify(text)
    routing = ROUTING.get(result["category"], ROUTING["General Inquiry"])
    return {
        "input": text,
        "category":   result["category"],
        "confidence": result["confidence"],
        "reasoning":  result["reasoning"],
        **routing,
    }


def print_result(result: dict) -> None:
    """Pretty-print the inference result."""
    print("\n" + "─" * 60)
    print(f"  INPUT      : {result['input']}")
    print("─" * 60)
    print(f"  CATEGORY   : {result['category']}")
    print(f"  CONFIDENCE : {result['confidence']:.0%}")
    print(f"  REASONING  : {result['reasoning']}")
    print("─" * 60)
    print(f"  ROUTED TO  : {result['department']}")
    print(f"  PRIORITY   : {result['priority']}")
    print(f"  NEXT STEP  : {result['action']}")
    print("─" * 60 + "\n")


# ── CLI entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    classifier = HOAClassifier()

    if len(sys.argv) > 1:
        # Single request passed as command-line argument
        text = " ".join(sys.argv[1:])
        result = process_request(text, classifier)
        print_result(result)
    else:
        # Interactive mode
        print("\nHOA Task Auto — Interactive Inference")
        print("Type a resident request and press Enter. Type 'quit' to exit.\n")
        while True:
            text = input("Resident request > ").strip()
            if text.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if not text:
                continue
            result = process_request(text, classifier)
            print_result(result)
