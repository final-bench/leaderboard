# FINAL Bench: Functional Metacognitive Reasoning Benchmark

> **"Not how much AI knows — but whether it knows what it doesn't know, and can fix it."**

---

## Overview

**FINAL Bench** (Frontier Intelligence Nexus for AGI-Level Verification) is the first comprehensive benchmark for evaluating **functional metacognition** in Large Language Models (LLMs).

Unlike existing benchmarks (MMLU, HumanEval, GPQA) that measure only final-answer accuracy, FINAL Bench evaluates the **entire pipeline of error detection, acknowledgment, and correction** — the hallmark of expert-level intelligence and a prerequisite for AGI.

| Item | Detail |
|------|--------|
| **Version** | 3.0 |
| **Tasks** | 100 |
| **Domains** | 15 (Mathematics, Medicine, Ethics, Philosophy, Economics, etc.) |
| **Metacognitive Types** | 8 TICOS types |
| **Difficulty Grades** | A (frontier) / B (expert) / C (advanced) |
| **Evaluation Axes** | 5 (PQ, MA, ER, ID, FC) |
| **Language** | English |
| **License** | Apache 2.0 |

---

## Why FINAL Bench?

### Metacognition Is the Gateway to AGI

Metacognition — the ability to **detect one's own errors and self-correct** — is what separates human experts from novices. Without this capability, no system can achieve AGI regardless of its knowledge breadth or reasoning depth.

### Limitations of Existing Benchmarks

| Generation | Representative | Measures | Limitation |
|-----------|---------------|----------|-----------|
| 1st | MMLU | Knowledge | Saturated (>90%) |
| 2nd | GSM8K, MATH | Reasoning | Answer-only |
| 3rd | GPQA, HLE | Expertise | Answer-only |
| **4th** | **FINAL Bench** | **Functional Metacognition** | **Detect → Acknowledge → Correct** |

### Key Findings (9 SOTA Models Evaluated)

Evaluation of 9 state-of-the-art models (GPT-5.2, Claude Opus 4.6, Gemini 3 Pro, DeepSeek-V3.2, and others) reveals:

- **ER Dominance**: **94.8%** of MetaCog gain originates from the Error Recovery axis alone
- **Declarative-Procedural Gap**: All 9 models can *verbalize* uncertainty but cannot *act* on it — mean MA–ER gap of 0.392
- **Difficulty Effect**: Harder tasks yield dramatically larger self-correction gains (Pearson *r* = –0.777, *p* < 0.001)

---

## Dataset Structure

### Task Fields

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | string | Unique identifier (e.g., FINAL-A01, FINAL-B15) |
| `domain` | string | One of 15 domains |
| `grade` | string | Difficulty grade: A / B / C |
| `ticos_type` | string | One of 8 metacognitive types |
| `difficulty` | string | frontier / expert |
| `lens` | string | Evaluation lens (theoretical / quantitative / debate) |
| `title` | string | Task title |
| `prompt` | string | Full prompt presented to the model |
| `expected_behavior` | string | Description of ideal metacognitive behavior |
| `hidden_trap` | string | Description of the embedded cognitive trap |
| `ticos_required` | string | Required TICOS elements (comma-separated) |
| `ticos_optional` | string | Optional TICOS elements (comma-separated) |

### Grade Distribution

| Grade | Tasks | Weight | Characteristics |
|-------|-------|--------|----------------|
| **A** (frontier) | 50 | ×1.5 | Open problems, multi-stage traps |
| **B** (expert) | 33 | ×1.0 | Expert-level with embedded reversals |
| **C** (advanced) | 17 | ×0.7 | Advanced undergraduate level |

### Domain Distribution (15 domains)

| Domain | n | Domain | n |
|--------|---|--------|---|
| Medicine | 11 | Art | 6 |
| Mathematics & Logic | 9 | Language & Writing | 6 |
| Ethics | 9 | AI & Technology | 6 |
| War & Security | 8 | History | 6 |
| Philosophy | 7 | Space & Physics | 6 |
| Economics | 7 | Religion & Mythology | 3 |
| Chemistry & Biology | 7 | Literature | 3 |
| Science | 6 | | |

### TICOS Metacognitive Type Distribution (8 types)

| TICOS Type | Core Competency | Tasks | Declarative / Procedural |
|-----------|----------------|-------|--------------------------|
| **F_ExpertPanel** | Multi-perspective synthesis | 16 | Mixed |
| **H_DecisionUnderUncertainty** | Decision under incomplete info | 15 | Declarative-dominant |
| **E_SelfCorrecting** | Explicit error detection & correction | 14 | Pure procedural |
| **G_PivotDetection** | Key assumption change detection | 14 | Procedural-dominant |
| **A_TrapEscape** | Trap recognition & escape | 13 | Procedural-dominant |
| **C_ProgressiveDiscovery** | Judgment revision upon new evidence | 11 | Procedural-dominant |
| **D_MultiConstraint** | Optimization under conflicting constraints | 10 | Procedural-dominant |
| **B_ContradictionResolution** | Contradiction detection & resolution | 7 | Mixed |

---

## Five-Axis Evaluation Rubric

Each task is independently scored on five axes:

| Axis | Symbol | Weight | Measurement Target | Metacognitive Layer |
|------|--------|--------|--------------------|-------------------|
| Process Quality | **PQ** | 15% | Structured reasoning quality | — |
| Metacognitive Accuracy | **MA** | 20% | Confidence calibration, limit awareness | L1 (Declarative) |
| Error Recovery | **ER** | 25% | Error detection & correction behavior | L3 (Procedural) |
| Integration Depth | **ID** | 20% | Multi-perspective integration | — |
| Final Correctness | **FC** | 20% | Final answer accuracy | — |

**FINAL Score** = Σ(weighted_score × grade_weight) / Σ(grade_weight)

### The MA–ER Separation: Core Innovation

- **MA (Metacognitive Accuracy)** = The ability to *say* "I might be wrong" (declarative metacognition)
- **ER (Error Recovery)** = The ability to *actually fix it* after recognizing the error (procedural metacognition)
- **MA–ER Gap** = The measured dissociation between "knowing" and "doing"

This separation directly maps to the monitoring–control model of Nelson & Narens (1990) from cognitive psychology.

---

## Usage

### Loading the Dataset

```python
from datasets import load_dataset

dataset = load_dataset("FINAL-Bench/Metacognitive", split="train")

# Total 100 tasks
print(f"Total tasks: {len(dataset)}")

# Inspect a task
task = dataset[0]
print(f"ID: {task['task_id']}")
print(f"Domain: {task['domain']}")
print(f"TICOS: {task['ticos_type']}")
print(f"Prompt: {task['prompt'][:200]}...")
```

### Baseline Evaluation (Single API Call)

```python
def evaluate_baseline(task, client, model_name):
    """Baseline condition: single call, no self-correction prompting."""
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": task['prompt']}],
        temperature=0.0
    )
    return response.choices[0].message.content

results = []
for task in dataset:
    response = evaluate_baseline(task, client, "your-model")
    results.append({
        "task_id": task['task_id'],
        "response": response
    })
```

### Five-Axis Judge Evaluation

```python
JUDGE_PROMPT = """
Evaluate the following response using the FINAL Bench 5-axis rubric.

[Task]
{prompt}

[Expected Behavior]
{expected_behavior}

[Hidden Trap]
{hidden_trap}

[Model Response]
{response}

Score each axis from 0.00 to 1.00 (in 0.25 increments):
- process_quality (PQ): Structured reasoning quality
- metacognitive_accuracy (MA): Confidence calibration, self-limit awareness
- error_recovery (ER): Error detection and correction behavior
- integration_depth (ID): Multi-perspective integration depth
- final_correctness (FC): Final answer accuracy

Output in JSON format.
"""
```

---

## Benchmark Results (9 SOTA Models)

### Key Findings — Visual Summary

![Fig 1. Multi-Model Leaderboard](fig1.png)
*Figure 1. Baseline + MetaCog scores and MetaCog gain (Δ_MC) across 9 models.*

![Fig 2. ER Transformation](fig2.png)
*Figure 2. Error Recovery distribution shift — 79.6% at floor (Baseline) → 98.1% at ≥0.75 (MetaCog).*

![Fig 3. Declarative-Procedural Gap](fig3.png)
*Figure 3. MA vs ER scatter plot showing the Baseline (○) → MetaCog (□) transition for all 9 models.*

![Fig 4. Difficulty Effect](fig4.png)
*Figure 4. Harder tasks benefit more from MetaCog (Pearson r = –0.777, p < 0.001).*

![Fig 5. Five-Axis Contribution](fig5.png)
*Figure 5. ER accounts for 94.8% of the total MetaCog gain across 9 models.*

### Baseline Leaderboard

| Rank | Model | FINAL | PQ | MA | ER | ID | FC | MA–ER Gap |
|------|-------|-------|----|----|----|----|-----|-----------|
| 1 | Kimi K2.5 | 68.71 | 0.775 | 0.775 | 0.450 | 0.767 | 0.750 | 0.325 |
| 2 | GPT-5.2 | 62.76 | 0.750 | 0.750 | 0.336 | 0.724 | 0.681 | 0.414 |
| 3 | GLM-5 | 62.50 | 0.750 | 0.750 | 0.284 | 0.733 | 0.724 | 0.466 |
| 4 | MiniMax-M1-2.5 | 60.54 | 0.742 | 0.733 | 0.250 | 0.725 | 0.700 | 0.483 |
| 5 | GPT-OSS-120B | 60.42 | 0.750 | 0.708 | 0.267 | 0.725 | 0.692 | 0.442 |
| 6 | DeepSeek-V3.2 | 60.04 | 0.750 | 0.700 | 0.258 | 0.683 | 0.733 | 0.442 |
| 7 | GLM-4.7P | 59.54 | 0.750 | 0.575 | 0.292 | 0.733 | 0.742 | 0.283 |
| 8 | Gemini 3 Pro | 59.50 | 0.750 | 0.550 | 0.317 | 0.750 | 0.717 | 0.233 |
| 9 | Claude Opus 4.6 | 56.04 | 0.692 | 0.708 | 0.267 | 0.725 | 0.517 | 0.442 |
| | **Mean** | **61.12** | **0.745** | **0.694** | **0.302** | **0.729** | **0.695** | **0.392** |

### MetaCog Leaderboard

| Rank | Model | FINAL | ER | Δ_MC |
|------|-------|-------|------|------|
| 1 | Kimi K2.5 | 78.54 | 0.908 | +9.83 |
| 2 | Gemini 3 Pro | 77.08 | 0.875 | +17.58 |
| 3 | GPT-5.2 | 76.50 | 0.792 | +13.74 |
| 4 | GLM-5 | 76.38 | 0.808 | +13.88 |
| 5 | Claude Opus 4.6 | 76.17 | 0.867 | +20.13 |
| | **Mean** | **75.17** | **0.835** | **+14.05** |

### Five-Axis Contribution Analysis

| Rubric | Contribution | Interpretation |
|--------|-------------|---------------|
| **Error Recovery** | **94.8%** | Nearly all of the self-correction effect |
| Metacognitive Accuracy | 5.0% | "Saying" ability barely changes |
| Remaining 3 axes | 0.2% | Negligible change |

---

## Theoretical Background

### Functional Metacognition

> **Definition.** Observable behavioral patterns in which a model *detects*, *acknowledges*, and *corrects* errors in its own reasoning. Whether this pattern shares the same internal mechanism as human subjective self-awareness is outside the scope of measurement; only behavioral indicators are assessed.

This definition is grounded in the functionalist tradition of Dennett (1987) and Block (1995), avoiding the anthropomorphic fallacy (Shanahan, 2024).

### Three-Layer Model of AI Metacognition

| Layer | Mechanism | FINAL Bench |
|-------|-----------|-------------|
| **L1** Surface self-reflection | Linguistic expressions ("I'm not certain...") | **Measured via MA rubric** |
| **L2** Embedding-space uncertainty | Logit entropy, OOD detection | Not measured (planned) |
| **L3** Behavioral self-correction | Error detection → reasoning revision | **Measured via ER rubric** |

### TICOS Framework

**T**ransparency · **I**ntrospection · **C**alibration · **O**bjectivity · **S**elf-correction

Each task is classified by a required/optional combination of these five metacognitive elements.

---

## Design Principles

### 1. Trap-Embedded Design
All 100 tasks contain hidden cognitive traps grounded in established cognitive biases — availability heuristic, confirmation bias, anchoring, base-rate neglect, and more. The benchmark measures the model's ability to "fall into and climb out of" these traps.

### 2. Declarative-Procedural Separation
MA and ER are scored as independent rubrics, enabling quantification of the gap between "the ability to say I don't know" and "the ability to actually fix it." No prior benchmark supports this distinction.

### 3. Comparative Condition Design
Baseline (single call) and MetaCog (self-correction scaffold) conditions isolate the causal effect of functional metacognition, following placebo-controlled clinical trial logic.

### 4. Anti-Contamination Design
All tasks were originally designed for FINAL Bench. They are not variants of existing benchmark problems and cannot be found in search engines or training data.

---

## Paper

**FINAL Bench: Measuring Functional Metacognitive Reasoning in Large Language Models**

Taebong Kim, Minsik Kim, Sunyoung Choi, Jaewon Jang

*Under review at a leading international AI venue.*

- [FINAL_Bench_paper.pdf](/FINAL_Bench_paper.pdf) — Full paper (under review)
- DOI: [10.57967/hf/7873](https://doi.org/10.57967/hf/7873)

---
## Application

- [Do Bubbles Form When Tens of Thousands of AIs Simulate Capitalism?] https://huggingface.co/spaces/Heartsync/Prompt-Dump
- [Article] https://huggingface.co/blog/FINAL-Bench/pumpdump


## Citation

```bibtex
@dataset{final_bench_2026,
  title={FINAL Bench: Measuring Functional Metacognitive Reasoning in Large Language Models},
  author={Kim, Taebong and Kim, Minsik and Choi, Sunyoung and Jang, Jaewon},
  year={2026},
  version={3.0},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/datasets/FINAL-Bench/Metacognitive}}
}
```

---

## License

This dataset is distributed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

- Academic and commercial use permitted
- Modification and redistribution permitted
- Attribution required

---

## Contact

- **Corresponding Author**: Taebong Kim (arxivgpt@gmail.com)
- **Affiliations**: VIDRAFT / Ginigen AI, Seoul, South Korea

---

## Acknowledgments

This benchmark is grounded in metacognition theory from cognitive psychology (Flavell, 1979; Nelson & Narens, 1990) and recent LLM self-correction research (DeepSeek-R1, Self-Correction Bench, ReMA). We thank all model providers whose systems were evaluated.
