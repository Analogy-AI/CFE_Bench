# CFE-Bench: Classroom Final Exam Benchmark

Large language models and vision–language models remain brittle on college-level STEM problems. Existing reasoning benchmarks often suffer from latent errors, ambiguous rubrics, subject narrowness, or saturation.

**CFE-Bench** (**C**lassroom **F**inal **E**xam) is a text-only and multimodal reasoning benchmark built from authentic, repeatedly used university homework and exam problems sourced from instructor-maintained course materials and verified by professors. It contains **305 text-only** and **144 multimodal** samples spanning **20+ subjects** across physics, mathematics, and other STEM domains. CFE-Bench introduces a variable-based verification protocol.

# Model Performance on CFE-Bench

We report two complementary accuracy metrics for both the text-only and multimodal subsets.

- **Variable Accuracy**: For each question containing multiple annotated variables, we compute the proportion of correctly extracted variables, then average this proportion across all questions.
- **Question Accuracy**: The proportion of questions for which all variables are correct.

🟢 = Open-weights models · 🟠 = Proprietary models

**Bold** = best in group · *Italic* = second-best in group

---

## Text Subset (305)

| Type | Model | Variable Accuracy | Question Accuracy |
|:---:|---|:---:|:---:|
| 🟢 | gemma-3-27b-it | 0.14 | 0.10 |
| 🟢 | Ministral-3-14B-Reasoning | 0.18 | 0.13 |
| 🟢 | Llama-4-Maverick | 0.25 | 0.20 |
| 🟢 | gpt-oss-120b | 0.41 | 0.34 |
| 🟢 | Qwen3-235B-Instruct | 0.37 | 0.32 |
| 🟢 | Qwen3-235B-Thinking | 0.39 | 0.33 |
| 🟢 | MiniMax-M2.1 | 0.33 | 0.28 |
| 🟢 | Kimi-K2-Instruct | 0.25 | 0.19 |
| 🟢 | Kimi-K2-Thinking | 0.46 | 0.39 |
| 🟢 | **Kimi-K2.5** | **0.51** | **0.44** |
| 🟢 | GLM-4.7 | 0.45 | 0.39 |
| 🟢 | GLM-5 | 0.47 | 0.41 |
| 🟢 | deepseek V3.2 (chat) | 0.48 | 0.42 |
| 🟢 | *deepseek V3.2 (reasoner)* | *0.50* | *0.43* |
| 🟠 | claude-sonnet-4.5 | 0.37 | 0.30 |
| 🟠 | claude-opus-4.5 | 0.49 | 0.42 |
| 🟠 | claude-opus-4.6 | 0.59 | 0.53 |
| 🟠 | grok-4-0709 | 0.53 | 0.48 |
| 🟠 | grok-4.1-fast-reasoning | 0.50 | 0.44 |
| 🟠 | gpt-5.2 | 0.58 | 0.51 |
| 🟠 | **gemini-3-flash-preview** | **0.66** | **0.59** |
| 🟠 | *gemini-3-pro-preview* | *0.65* | *0.58* |

---

## Multimodal Subset (144)

| Type | Model | Variable Accuracy | Question Accuracy |
|:---:|---|:---:|:---:|
| 🟢 | gemma-3-27b-it | 0.07 | 0.03 |
| 🟢 | *Llama-4-Maverick* | *0.16* | ***0.10*** |
| 🟢 | InternVL3-78B-Instruct | 0.07 | 0.03 |
| 🟢 | InternVL3.5-GPT-OSS-20B | 0.04 | 0.02 |
| 🟢 | InternVL3.5-241B-A28B | 0.11 | 0.05 |
| 🟢 | **Qwen3-VL-32B-Instruct** | **0.19** | **0.10** |
| 🟢 | GLM-4.6v | 0.15 | *0.08* |
| 🟠 | qvq-max | 0.10 | 0.06 |
| 🟠 | claude-sonnet-4.5 | 0.27 | 0.19 |
| 🟠 | claude-opus-4.5 | 0.38 | 0.31 |
| 🟠 | claude-opus-4.6 | 0.44 | 0.37 |
| 🟠 | grok-4-0709 | 0.36 | 0.29 |
| 🟠 | grok-4.1-fast-reasoning | 0.33 | 0.26 |
| 🟠 | gpt-5.2 | 0.51 | 0.44 |
| 🟠 | **gemini-3-flash** | **0.59** | **0.51** |
| 🟠 | *gemini-3-pro-preview* | *0.57* | *0.49* |

## Setup

```bash
git clone https://github.com/GCYZSL/CFE_Benchmark.git
cd CFE_Benchmark
conda create -n CFE python=3.10
conda activate CFE
pip install google-genai
pip install tqdm
pip install openai
```

## Dataset Structure

CFE-Bench contains two subsets:

| Subset | File | # Questions | Description |
|--------|------|-------------|-------------|
| Text-only | `CFE_text.json` | 305 | Pure text STEM problems |
| Multimodal | `CFE_mm.json` | 144 | Problems with diagrams, plots, symbolic notation |

### Subjects

Both subsets span multiple STEM domains including Physics, Mathematics, Electrical Engineering, Mechanical Engineering, Chemistry, Biology, Statistics, Computer Science, and others.

### Data Format

Each entry in the JSON files has the following structure:

```json
{
  "question": {
    "text": "Problem statement...",
    "images": []
  },
  "answer": {
    "text": "Full ground-truth solution...",
    "images": []
  },
  "id": "unique_hash_id",
  "short_answer_value": ["6250 kbp"],
  "short_answer_variable": ["genome_size"],
  "short_answer_description": [
    "The estimated total size of the bacterial genome..."
  ],
  "short_answer_type": ["other"],
  "reasoning_flow": [
    {
      "step_id": 1,
      "step": "Sub-question for this reasoning step",
      "verifiable_answer": 0.008
    }
  ]
}
```

### Key Fields

| Field | Description |
|-------|-------------|
| `question.text` | The full problem statement |
| `question.images` | Associated images (diagrams, plots) for multimodal problems |
| `answer.text` | Complete ground-truth solution with derivation |
| `short_answer_value` | List of verifiable target values (V_gt) |
| `short_answer_variable` | Variable names corresponding to each target value |
| `short_answer_description` | Semantic descriptions to guide variable extraction |
| `short_answer_type` | Type of each answer (numeric, formula, other) |
| `reasoning_flow` | Ordered list of reasoning units, each with a sub-question (`step`) and a `verifiable_answer` |

## Usage

Evaluation is a two-step pipeline: **(1)** generate model responses, then **(2)** evaluate them with an LLM judge.

### Step 1: Generate Responses

`generate_responses.py` runs the test model on the benchmark and saves generated answers.

```bash
python generate_responses.py \
  --input CFE_text.json \
  --test_provider gemini \
  --test_model gemini-3-flash-preview \
  --test_api_key YOUR_GEMINI_API_KEY \
  --k 1 \
  --max_workers 2
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | `CFE_text.json` | Input benchmark JSON file (`CFE_text.json` or `CFE_mm.json`) |
| `--image_folder` | `./` | Path to the folder containing images (for multimodal subset) |
| `--output` | auto-generated | Output JSON file path. If not set, defaults to `{input}_k_{k}_test_{model}.json` |
| `--k` | `1` | Number of answers to generate per question (k in pass@k) |
| `--test_provider` | `gemini` | Model provider |
| `--test_model` | `gemini-3-flash-preview` | Model name |
| `--test_api_key` | — | API key for the test model |
| `--max_workers` | `2` | Number of parallel workers (be careful with rate limits) |

**Output format:** A JSON list where each entry contains:

```json
{
  "generated_answers": ["model response 1", "model response 2", ...],
  "id": "matching_question_id"
}
```

The number of responses per question equals `k`.

### Step 2: Evaluate Responses

`evaluation.py` uses an LLM judge (OpenAI) to extract variable values from model responses and verify them against ground truth.

```bash
python evaluation.py \
  --input CFE_text_k_1_test_gemini-3-flash-preview.json \
  --input_benchmark CFE_text.json \
  --judge_provider openai \
  --judge_model gpt-5-mini-2025-08-07 \
  --judge_api_key YOUR_OPENAI_API_KEY \
  --k 1 \
  --max_workers 2
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | — | Generated responses JSON from Step 1 |
| `--input_benchmark` | `CFE_text.json` | Original benchmark JSON file (needed for ground-truth answers) |
| `--image_folder` | `./` | Path to image folder (for multimodal) |
| `--output` | auto-generated | Output JSON file path |
| `--k` | `1` | k value for pass@k (must match Step 1) |
| `--judge_provider` | `openai` | Judge model provider |
| `--judge_model` | `gpt-5-mini-2025-08-07` | Judge model name |
| `--judge_api_key` | — | API key for the judge model |
| `--max_workers` | `2` | Number of parallel workers |

**Evaluation pipeline:** For each generated response, the judge performs two steps:

1. **Value Extraction** — Extracts variable values from the model response using the variable names, types, and descriptions defined in the benchmark.
2. **Verification** — Compares each extracted value against the ground-truth value, handling mathematical equivalence, unit conversions, rounding tolerance, and format differences.

**Output files:**

1. **Detailed results** (`{input}_k_{k}_test_{judge_model}.json`): Full per-sample results including extracted values, per-variable judgments, and pass@k scores.
2. **Summary** (`..._summary.json`): Aggregate metrics and configuration.

**Output metrics printed to console:**

```
==================================================
PASS@K EVALUATION RESULTS
==================================================
Pass@1: XX.XX%
Total samples: N
overall_avg_variable_accuracy: X.XX
overall_question_accuracy: X.XX
==================================================
```

## Metrics

| Metric | Description |
|--------|-------------|
| **pass@k** | Probability that at least one of k generated answers is correct, using the unbiased estimator: `1 - C(n-c, k) / C(n, k)` where n = total samples, c = correct samples |
| **overall_question_accuracy** | Fraction of generated answers that are fully correct (all variables correct), averaged across all questions |
| **overall_avg_variable_accuracy** | Average per-variable accuracy across all questions and all generated answers. For questions with multiple variables, this measures partial credit at the variable level |

## Evaluation Methodology: Variable-Based Verification

CFE-Bench defines ground-truth variables **V_gt = {(v₁, d₁, x₁), ..., (vₙ, dₙ, xₙ)}** where each tuple contains a variable name, semantic description, and target value.

It extracts specific variable values from the model's response using the variable names and descriptions, then compares against V_gt. This achieves the highest accuracy and significantly reduces false positives by isolating specific variables.

## Citation

```bibtex
@article{CFE2025,
  title={CFE-Bench: Classroom Final Exam Benchmark},
  year={2025}
}
```

## Acknowledgments

We thank the Analogy AI staffs. instructors and course staff who made their materials publicly available, enabling the construction of this benchmark.