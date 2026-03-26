"""
model.py
──────────────────────────────────────────────────────────────────────────────
HOA Task Auto — Claude Sonnet 4.6 model wrapper.
Fixed version: more explicit prompt + robust JSON parsing + debug mode.
──────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import re
import anthropic
import yaml
from dotenv import load_dotenv

load_dotenv()

CATEGORIES = [
    "Maintenance Request",
    "Dues & Payment",
    "Rule Violation Report",
    "Amenity Booking",
    "General Inquiry",
    "Complaint / Dispute",
]

# ── Few-shot examples ─────────────────────────────────────────────────────────
FEW_SHOT_EXAMPLES = [
    ("The street light outside unit 204 has been out for a week.",        "Maintenance Request"),
    ("Pool pump at amenity area 3 stopped working.",                       "Maintenance Request"),
    ("I would like to set up automatic payment for my monthly dues.",      "Dues & Payment"),
    ("Can I get a receipt for my last payment of unit 512?",               "Dues & Payment"),
    ("My neighbor at unit 318 is parking in a handicap spot.",             "Rule Violation Report"),
    ("Loud music from unit 101 after quiet hours.",                        "Rule Violation Report"),
    ("I'd like to reserve the clubhouse for a birthday party Saturday.",   "Amenity Booking"),
    ("Can I book the BBQ area near pool 2 for this weekend?",              "Amenity Booking"),
    ("What are the office hours for the HOA management team?",             "General Inquiry"),
    ("How do I submit a suggestion to the HOA board?",                     "General Inquiry"),
    ("I am disputing the fine I received for unit 405 last month.",        "Complaint / Dispute"),
    ("I feel the recent rule change was not communicated properly.",       "Complaint / Dispute"),
]

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an HOA request classifier. Classify resident messages into exactly one category.

The 6 valid categories are:
- Maintenance Request
- Dues & Payment
- Rule Violation Report
- Amenity Booking
- General Inquiry
- Complaint / Dispute

You MUST respond with ONLY a JSON object. No explanation before or after. No markdown. No code blocks.
The JSON must have exactly these three keys:
{
  "category": "<exact category name from the list above>",
  "confidence": <number between 0.0 and 1.0>,
  "reasoning": "<one short sentence>"
}"""


def _build_user_message(text: str) -> str:
    """Build the user message with few-shot examples."""
    lines = ["Examples:\n"]
    for msg, label in FEW_SHOT_EXAMPLES:
        lines.append(f'Input: "{msg}"')
        lines.append(f'Output: {{"category": "{label}", "confidence": 0.99, "reasoning": "Matches {label} pattern."}}\n')
    lines.append(f'Now classify this:\nInput: "{text}"')
    lines.append("Output:")
    return "\n".join(lines)


def _parse_response(raw: str) -> dict:
    """Robustly extract JSON from Claude response."""
    # 1. Direct parse
    try:
        return json.loads(raw.strip())
    except Exception:
        pass
    # 2. Strip markdown fences
    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    # 3. Extract first {...} block
    match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    # 4. Keyword fallback
    raw_lower = raw.lower()
    for cat in CATEGORIES:
        if cat.lower() in raw_lower:
            return {"category": cat, "confidence": 0.7,
                    "reasoning": "Extracted via keyword match fallback."}
    # 5. Total fallback
    return {"category": "General Inquiry", "confidence": 0.1,
            "reasoning": "Could not parse response."}


def _validate_category(category: str) -> str:
    """Ensure category is one of the 6 valid ones."""
    if category in CATEGORIES:
        return category
    cat_lower = category.lower()
    for cat in CATEGORIES:
        if cat.lower() in cat_lower or cat_lower in cat.lower():
            return cat
    for cat in CATEGORIES:
        for word in cat.lower().split():
            if word in cat_lower and len(word) > 4:
                return cat
    return "General Inquiry"


class HOAClassifier:
    """Wrapper around Claude Sonnet 4.6 for HOA request classification."""

    def __init__(self, config_path: str = "config.yaml", debug: bool = False):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.model       = cfg["api"]["model"]
        self.max_tokens  = cfg["api"]["max_tokens"]
        self.temperature = cfg["api"]["temperature"]
        self.debug       = debug

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set.")
        self.client = anthropic.Anthropic(api_key=api_key)

    def classify(self, text: str) -> dict:
        """Classify a single HOA request. Returns {category, confidence, reasoning}"""
        user_msg = _build_user_message(text)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = response.content[0].text.strip()
        if self.debug:
            print(f"\n[DEBUG] Raw API response:\n{raw}\n")
        result = _parse_response(raw)
        result["category"] = _validate_category(result.get("category", ""))
        return result


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running smoke test on 6 samples...\n")
    classifier = HOAClassifier(debug=True)
    samples = [
        ("The elevator in block 3 is making a loud noise.",          "Maintenance Request"),
        ("I need to update my payment method on file.",               "Dues & Payment"),
        ("My neighbor installed an unauthorized satellite dish.",     "Rule Violation Report"),
        ("I'd like to book the pool area for Sunday afternoon.",      "Amenity Booking"),
        ("What is the process for getting a parking permit?",         "General Inquiry"),
        ("I want to appeal the architectural review board decision.", "Complaint / Dispute"),
    ]
    correct = 0
    for text, expected in samples:
        result = classifier.classify(text)
        match = "✅" if result["category"] == expected else "❌"
        print(f"{match} Expected : {expected}")
        print(f"   Got      : {result['category']}  ({result['confidence']:.0%})")
        print(f"   Text     : {text[:65]}")
        print()
        if result["category"] == expected:
            correct += 1
    print(f"Smoke test: {correct}/{len(samples)} correct  ({correct/len(samples):.0%})")
