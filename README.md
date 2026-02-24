# CFE-Bench: Classroom Final Exam Benchmark

Large language models and vision–language models remain brittle on college-level STEM problems. Existing reasoning benchmarks often suffer from latent errors, ambiguous rubrics, subject narrowness, or saturation.

**CFE-Bench** (**C**lassroom **F**inal **E**xam) is a text-only and multimodal reasoning benchmark built from authentic, repeatedly used university homework and exam problems sourced from instructor-maintained course materials and verified by professors. It contains **305 text-only** and **144 multimodal** samples spanning **20+ subjects** across physics, mathematics, and other STEM domains. CFE-Bench introduces a variable-based verification protocol.

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
@misc{gao2026classroomfinalexaminstructortested,
      title={Classroom Final Exam: An Instructor-Tested Reasoning Benchmark}, 
      author={Chongyang Gao and Diji Yang and Shuyan Zhou and Xichen Yan and Luchuan Song and Shuo Li and Kezhen Chen},
      year={2026},
      eprint={2602.19517},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.19517}, 
}
```

## Acknowledgments

We thank the Analogy AI staffs. instructors and course staff who made their materials publicly available, enabling the construction of this benchmark.