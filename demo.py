"""
demo.py
──────────────────────────────────────────────────────────────────────────────
HOA Task Auto — Gradio interactive demo.

Usage:
    python demo.py

Opens a local web UI at http://localhost:7860
──────────────────────────────────────────────────────────────────────────────
"""

import gradio as gr
import yaml
from model import HOAClassifier
from inference import process_request, ROUTING

# ── Load config & classifier ──────────────────────────────────────────────────
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

classifier = HOAClassifier()

# ── Category colour map for display ──────────────────────────────────────────
CATEGORY_EMOJI = {
    "Maintenance Request":   "🔧",
    "Dues & Payment":        "💳",
    "Rule Violation Report": "🚨",
    "Amenity Booking":       "📅",
    "General Inquiry":       "❓",
    "Complaint / Dispute":   "⚖️",
}

EXAMPLE_REQUESTS = [
    "The street light outside unit 204 has been out for a week.",
    "I would like to set up automatic payment for my monthly dues.",
    "My neighbor at unit 318 is parking in a handicap spot without a permit.",
    "I'd like to reserve the clubhouse for a birthday party on Saturday.",
    "What are the office hours for the HOA management team?",
    "I am disputing the fine I received for unit 405 last month.",
]


def classify_request(text: str):
    """Gradio handler — returns formatted result strings."""
    if not text.strip():
        return "⚠ Please enter a request.", "", "", "", ""

    try:
        result = process_request(text, classifier)
    except Exception as e:
        return f"❌ Error: {e}", "", "", "", ""

    emoji = CATEGORY_EMOJI.get(result["category"], "📋")
    category_str  = f"{emoji}  {result['category']}"
    confidence_str = f"{result['confidence']:.0%}"
    reasoning_str  = result["reasoning"]
    routing_str    = (
        f"**Department:** {result['department']}\n"
        f"**Priority:** {result['priority']}\n"
        f"**Next Step:** {result['action']}"
    )

    return category_str, confidence_str, reasoning_str, routing_str


# ── Build Gradio UI ───────────────────────────────────────────────────────────
with gr.Blocks(
    title="HOA Task Auto",
    theme=gr.themes.Base(
        primary_hue="blue",
        secondary_hue="orange",
    ),
) as demo:
    gr.Markdown(
        """
        # 🏘️ HOA Task Auto
        ### AI-Powered HOA Request Processing with Claude Sonnet 4.6
        Enter a resident request below and the system will automatically classify
        it into one of **6 task categories** and route it to the correct department.
        ---
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            input_box = gr.Textbox(
                label="Resident Request",
                placeholder="e.g. The elevator in block 3 is making a loud noise.",
                lines=4,
            )
            submit_btn = gr.Button("🚀 Process Request", variant="primary")

            gr.Examples(
                examples=EXAMPLE_REQUESTS,
                inputs=input_box,
                label="Example Requests",
            )

        with gr.Column(scale=2):
            category_out    = gr.Textbox(label="📋 Category",   interactive=False)
            confidence_out  = gr.Textbox(label="🎯 Confidence", interactive=False)
            reasoning_out   = gr.Textbox(label="💡 Reasoning",  interactive=False, lines=2)
            routing_out     = gr.Markdown(label="📬 Routing")

    submit_btn.click(
        fn=classify_request,
        inputs=input_box,
        outputs=[category_out, confidence_out, reasoning_out, routing_out],
    )
    input_box.submit(
        fn=classify_request,
        inputs=input_box,
        outputs=[category_out, confidence_out, reasoning_out, routing_out],
    )

    gr.Markdown(
        """
        ---
        **Course:** Deep Learning and AI — UCSC EXT AISV.X401(23)  
        **Model:** Claude Sonnet 4.6 (Anthropic) via API  
        **Presenter:** Yun Yan
        """
    )


# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port  = cfg["demo"]["port"]
    share = cfg["demo"]["share"]
    demo.launch(server_port=port, share=share)
