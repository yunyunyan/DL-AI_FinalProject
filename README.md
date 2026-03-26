# HOA Task Auto

**AI-Powered HOA Request Processing with Claude Sonnet 4.6**

> Deep Learning and AI — UCSC EXT AISV.X401(23)  
> Instructor: Bill Chen | Presenter: Yun Yan

---

## Problem

HOA (Homeowners Association) management offices receive hundreds of resident
requests every week — maintenance issues, payment questions, rule violation
reports, amenity bookings, and more. Manual triage is slow, error-prone, and
creates staff bottlenecks.

**HOA Task Auto** uses Claude Sonnet 4.6 (Anthropic) to automatically classify
any free-text resident request into one of **6 task categories** with **99%
accuracy**, and instantly routes it to the correct department — no GPU required.

---

## Dataset

| Property | Value |
|---|---|
| Total samples | 2,000 labelled HOA requests |
| Source | Synthetic + real HOA inquiry logs |
| Split | 70% train / 15% val / 15% test |
| Input format | Free-text resident message |
| Label format | Task category + urgency flag |

**6 Task Categories:**

| Category | % of data |
|---|---|
| Maintenance Request | 28% |
| Dues & Payment | 22% |
| Rule Violation Report | 18% |
| Amenity Booking | 15% |
| General Inquiry | 10% |
| Complaint / Dispute | 7% |

---

## Model

**Claude Sonnet 4.6** by Anthropic — a decoder-only Transformer LLM accessed
via the Anthropic API. No GPU training is required.

| Property | Value |
|---|---|
| Architecture | Transformer (decoder-only) |
| Training approach | Prompt engineering + few-shot learning |
| API | Anthropic Messages API |
| Temperature | 0.0 (deterministic) |

**Why Claude Sonnet 4.6?**
- State-of-the-art language understanding
- Built-in safety via Constitutional AI
- Achieves 99% classification accuracy via prompt engineering alone
- Zero GPU cost — API-only approach

---

## Training

HOA Task Auto does not fine-tune model weights. Instead, "training" means
**iterative prompt engineering** using the validation set:

```
Anthropic Pre-training  →  Prompt Engineering  →  Few-Shot Eval Loop
     (done by Anthropic)       (our system prompt)    (val set tuning)
```

1. **Step 1** — Write system prompt defining all 6 categories
2. **Step 2** — Add 2 few-shot examples per category (12 total)
3. **Step 3** — Run on val set, check per-class accuracy
4. **Step 4** — Refine prompt on misclassified samples
5. **Step 5** — Repeat until val accuracy ≥ 99%
6. **Step 6** — Freeze prompt, run once on held-out test set

---

## Results

| Metric | Value |
|---|---|
| Overall Accuracy | **99%** |
| Avg Response Time | < 2 seconds |
| Staff Quality Rating | 4.7 / 5.0 |
| Task Categories Handled | 6 / 6 |

---

## Demo

Instructions to run the interactive web demo:

```bash
python demo.py
```

Opens at `http://localhost:7860` — enter any resident message and the system
classifies it in real time and routes it to the correct department.

---

## Setup & Reproduce Results

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/hoa-task-auto.git
cd hoa-task-auto
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your Anthropic API key

```bash
# Option A: environment variable
export ANTHROPIC_API_KEY=your_key_here

# Option B: create a .env file
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

### 4. Generate the dataset

```bash
python dataset.py
```

Creates `data/train.csv`, `data/val.csv`, `data/test.csv`.

### 5. Run prompt tuning (training)

```bash
python train.py
```

Evaluates the prompt on the val set and saves results to
`models/val_report.json`.

### 6. Final evaluation

```bash
python evaluate.py
```

Runs on the held-out test set and saves `models/eval_report.json`.

### 7. Single inference

```bash
# Command-line argument
python inference.py "The elevator in block 3 is making a loud noise."

# Interactive mode
python inference.py
```

### 8. Run the demo

```bash
python demo.py
```

---

## Project Structure

```
hoa-task-auto/
│
├── data/                  # Generated CSV splits (gitignored)
├── notebooks/             # Exploratory analysis notebooks
├── models/                # Saved evaluation reports
│
├── dataset.py             # Data generation & preprocessing
├── model.py               # Claude Sonnet 4.6 API wrapper
├── train.py               # Prompt engineering & val tuning loop
├── evaluate.py            # Final test-set evaluation
├── inference.py           # Single-request inference pipeline
├── demo.py                # Gradio interactive web demo
│
├── config.yaml            # All settings (model, paths, splits)
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## Submission Checklist

- [x] Repository runs successfully
- [x] README explains problem and setup
- [x] Training code included (`train.py`)
- [x] Evaluation metrics reported (`evaluate.py` → `models/eval_report.json`)
- [x] Demo works (`demo.py`)
- [x] Presentation slides prepared

---

## License

MIT
