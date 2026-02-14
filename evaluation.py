import base64
import argparse
import json
import os
from typing import Union, List, Dict
from tqdm import tqdm
from pydantic import BaseModel
import time
import concurrent.futures
from functools import partial
from collections import defaultdict
from openai import OpenAI
from datetime import datetime
import re
from pathlib import Path


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class ResponseValueExtraction(BaseModel):
    extract_short_answer_value_list: list[Union[str, int, float, bool, None]]

class LLMJudgeVerification(BaseModel):
    is_correct: bool
    reasoning: str
    confidence: str  # "high", "medium", "low"

# llm judge extract value
RESPONSE_VALUE_EXTRACTION_PROMPT_WITH_TYPES = """
You are a strict data extraction engine. Extract values for `{variable_list}` from `{response}` based on `{type_list}` and '{reference_description}'.

### CRITICAL: SOURCE & CARDINALITY
1. **SOURCE OF TRUTH:** Extract values ONLY from `{response}`.
2. **ONE-TO-ONE MAPPING:** The output list MUST have the exact same number of elements as `{variable_list}`.
3. **ORDER:** Value at index `i` corresponds strictly to variable at index `i`.

### EXTRACTION RULES
- **Numeric:** Extract pure numbers only (remove units).
- **Formula:** Wrap LaTeX in `$` symbols.
- **Other:** Extract exact text with formatting.
- **Missing Data:** If a variable is not found in `{response}`, the value is "null".
    - *Constraint:* Do NOT collapse the list.
    - *Example:* Variables=["x", "y", "z"] where "y" and "z" are missing -> Output=[value_x, "null", "null"]

### INPUT DATA
**Question:** {question}
**Variables:** {variable_list}
**Types:** {type_list}
**Reference description:** {reference_description}
**Response (Extract Values):** {response}

### OUTPUT
Return a valid JSON object:
{{
    "short_answer_value_list": [value1, value2, ...]
}}
"""

# Judge prompt template
LLM_JUDGE_VERIFICATION_PROMPT = """
You are an expert mathematical and logical evaluator. Your task is to determine if an extracted answer matches the reference (ground truth) answer.

You will be given:
1. `variable_name`: The name of the variable being checked
2. `reference_value`: The ground truth answer (CORRECT)
3. `extracted_value`: The answer extracted from a model's response (TO BE VERIFIED)
4. `question_context` (optional): Context from the original question

### Your Task:

Determine if the `extracted_value` is mathematically/logically equivalent to the `reference_value`.

### Evaluation Rules:

**1. Mathematical Equivalence:**
- Check if values are mathematically equal, even if formatted differently
- Examples of CORRECT matches:
  - Reference: 5, Extracted: 5.0 → CORRECT
  - Reference: "1/2", Extracted: "0.5" → CORRECT
  - Reference: "50%", Extracted: "0.5" → CORRECT
  - Reference: "2²", Extracted: "4" → CORRECT
  - Reference: "√16", Extracted: "4" → CORRECT

**2. Unit Handling:**
- Values with the same quantity but different units may be correct if equivalent
- Examples:
  - Reference: "1 m", Extracted: "100 cm" → CORRECT
  - Reference: "1 kg", Extracted: "1000 g" → CORRECT
  - Reference: "60 mph", Extracted: "96.56 km/h" → CORRECT (if approximately equal)
- If units are fundamentally different (e.g., meters vs seconds), mark as INCORRECT

**3. Rounding and Precision:**
- Allow for reasonable rounding differences
- Examples of CORRECT matches:
  - Reference: 3.14159, Extracted: 3.14 → CORRECT
  - Reference: "5 m/s", Extracted: "5.0 m/s" → CORRECT
  - Reference: 10, Extracted: 10.001 → CORRECT (negligible difference)
- Large differences are INCORRECT:
  - Reference: 10, Extracted: 15 → INCORRECT

**4. String and Format Differences:**
- Ignore minor formatting differences if the meaning is the same
- Examples of CORRECT matches:
  - Reference: "x = 5", Extracted: "5" → CORRECT
  - Reference: "B", Extracted: "option B" → CORRECT
  - Reference: "true", Extracted: true → CORRECT
  - Reference: "E = mc²", Extracted: "E=mc^2" → CORRECT

**5. Logical Equivalence:**
- For boolean/true-false answers, check logical equivalence
- For multiple choice, check if the same option is selected

**6. Null Handling:**
- If both are null/None: CORRECT
- If only one is null: INCORRECT (unless the other is "none", "null", "N/A", etc.)

**7. Special Cases:**
- Mathematical expressions: Check if they simplify to the same value
- Vectors/coordinates: Check component-wise equality
- Sets: Order doesn't matter, content must match
- Fractions: Reduce to simplest form before comparing

### Confidence Levels:

- **high**: Clear match or clear mismatch, no ambiguity
- **medium**: Match with minor unit conversion or rounding involved
- **low**: Uncertain due to formatting ambiguity or complex expressions

### Output Format:

Return a JSON object with:
- `is_correct`: boolean (true if equivalent, false if not)
- `reasoning`: Detailed explanation of your decision
- `confidence`: "high", "medium", or "low"

### Examples:

**Example 1: Exact Match**
Variable: "x"
Reference: 5
Extracted: 5
Output:
{{
    "is_correct": true,
    "reasoning": "Both values are exactly 5, perfect match",
    "confidence": "high"
}}

**Example 2: Mathematical Equivalence**
Variable: "result"
Reference: "1/2"
Extracted: "0.5"
Output:
{{
    "is_correct": true,
    "reasoning": "1/2 equals 0.5, mathematically equivalent despite different representations",
    "confidence": "high"
}}

**Example 3: Unit Conversion**
Variable: "distance"
Reference: "1 m"
Extracted: "100 cm"
Output:
{{
    "is_correct": true,
    "reasoning": "1 meter equals 100 centimeters, correct unit conversion",
    "confidence": "high"
}}

**Example 4: Rounding Difference (Acceptable)**
Variable: "pi"
Reference: 3.14159
Extracted: 3.14
Output:
{{
    "is_correct": true,
    "reasoning": "Extracted value is a reasonable rounding of pi to 2 decimal places",
    "confidence": "medium"
}}

**Example 5: Wrong Answer**
Variable: "x"
Reference: 10
Extracted: 15
Output:
{{
    "is_correct": false,
    "reasoning": "10 does not equal 15, these are different values with no mathematical relationship",
    "confidence": "high"
}}

**Example 6: Format Difference (Still Correct)**
Variable: "answer"
Reference: "B"
Extracted: "option B"
Output:
{{
    "is_correct": true,
    "reasoning": "Both refer to option B, just different formatting",
    "confidence": "high"
}}

**Example 7: Both Null**
Variable: "result"
Reference: null
Extracted: null
Output:
{{
    "is_correct": true,
    "reasoning": "Both values are null, indicating no determinable answer in both cases",
    "confidence": "high"
}}

**Example 8: One Null**
Variable: "value"
Reference: 5
Extracted: null
Output:
{{
    "is_correct": false,
    "reasoning": "Reference has a value (5) but extracted is null, indicating the value was not found",
    "confidence": "high"
}}

**Example 9: Complex Expression**
Variable: "formula"
Reference: "E = mc²"
Extracted: "E=mc^2"
Output:
{{
    "is_correct": true,
    "reasoning": "Both represent Einstein's mass-energy equivalence, just different notation (² vs ^2)",
    "confidence": "high"
}}

**Example 10: Approximate Match**
Variable: "speed"
Reference: "60 mph"
Extracted: "96.5 km/h"
Output:
{{
    "is_correct": true,
    "reasoning": "60 mph equals approximately 96.56 km/h, extracted value is correct with minor rounding",
    "confidence": "medium"
}}

---

**Variable Name:**
{variable_name}

**Reference Value (Ground Truth):**
{reference_value}

**Extracted Value (To Verify):**
{extracted_value}

**Question Context:**
{question_context}
"""


class JudgeClient:
    def __init__(self, model_config):
        api_key = model_config["api_key"]
        self.provider = model_config["provider"]
        self.model_name = model_config["model"]
        # self.temperature = 0
        if model_config["provider"] == "openai":
            self.client = OpenAI(api_key=api_key)
        else:
            raise Exception("The provider is not supported")

    def chat(self, input, text_format):
        if self.provider == "openai":
            if "gpt-5" in self.model_name:
                response = self.client.responses.parse(
                    model=self.model_name,
                    input=input,
                    # temperature=self.temperature,
                    text_format=text_format,
                    reasoning={"effort": "low"},
                )
            else:
                response = self.client.responses.parse(
                    model=self.model_name,
                    input=input,
                    text_format=text_format,
                    # temperature=self.temperature,
                )
            return dict(response.output_parsed)
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": input}],
                temperature=self.temperature,
                top_k=1,
            )
            return response.choices[0].message.content


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate pass@k metric using the unbiased estimator.

    Args:
        n: Total number of samples generated
        c: Number of correct samples
        k: k value for pass@k (must be <= n)

    Returns:
        pass@k probability (0.0 to 1.0)

    Formula: pass@k = 1 - (n-c choose k) / (n choose k)

    This calculates the probability that at least one correct solution
    appears in k samples, given c correct solutions out of n total samples.
    """
    if n < k:
        return 0.0
    if c == 0:
        return 0.0
    if c >= k:
        return 1.0

    # Calculate using the unbiased estimator
    # pass@k = 1 - [(n-c)! * k! * (n-k)!] / [n! * (k-c)! * (n-c-k+c)!]
    # Simplified: 1 - prod((n-c-i)/(n-i) for i in range(k))

    def comb(n, k):
        """Calculate binomial coefficient n choose k"""
        if k > n or k < 0:
            return 0
        if k == 0 or k == n:
            return 1
        k = min(k, n - k)
        result = 1
        for i in range(k):
            result = result * (n - i) // (i + 1)
        return result

    numerator = comb(n - c, k)
    denominator = comb(n, k)

    if denominator == 0:
        return 0.0

    return 1.0 - (numerator / denominator)


def save_error_to_json(error_data, error_log_path: str = "./error_logs/") -> str:
    """
    Save error information to a JSON file.

    Args:
        error_data: Dictionary containing error information
        error_log_path: Directory path to save error logs

    Returns:
        Path to the saved error log file
    """
    # Create error log directory if it doesn't exist
    Path(error_log_path).mkdir(parents=True, exist_ok=True)

    # Generate timestamp-based filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"error_{timestamp}.json"
    filepath = Path(error_log_path) / filename

    # Add timestamp to error data
    error_data["timestamp"] = datetime.now().isoformat()

    # Save to JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(error_data, f, indent=2, ensure_ascii=False)

    return str(filepath)

def extract_values_from_response(client, question, variable_list, type_list,
                                 reference_description, response, id,
                                 retry_count: int = 10) -> Dict:
    """
    Extract values from a model response based on variable list and types.

    Returns:
        Dict containing extracted values
    """
    prompt = RESPONSE_VALUE_EXTRACTION_PROMPT_WITH_TYPES.format(
        question=question.get("text", question) if isinstance(question, dict) else question,
        variable_list=json.dumps(variable_list),
        type_list=json.dumps(type_list),
        reference_description=json.dumps(reference_description),
        response=response
    )


    input_content = prompt

    for attempt in range(retry_count):
        try:
            result = client.chat(input_content, ResponseValueExtraction)
            if len(variable_list) != len(result["extract_short_answer_value_list"]):
                raise InterruptedError
            return result
        except Exception as e:
            if attempt < retry_count - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                # Return null values for all variables on error
                error_info = {
                    "function": "extract_values_from_response",
                    "question": question.get("text", question) if isinstance(question, dict) else question,
                    "variable_list": variable_list,
                    "type_list": type_list,
                    "reference_description": reference_description,
                    "response": response,
                    "retry_count": retry_count,
                    "input_content": input_content,
                    "id": id,
                    "final_error": {
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    }
                }
                print(error_info)
                error_file = save_error_to_json(error_info, "./")
                print(f"Error logged to: {error_file}")

                return {
                    "extract_short_answer_value_list": ["null"] * len(variable_list),
                    "reasoning": f"Error during extraction: {str(e)}",
                    "error": str(e),
                    "error_log_file": error_file
                }


def verify_answer_with_llm_judge(client, variable_name: str, reference_value,
                                 extracted_value, question_data: str = "",
                                 retry_count: int = 3) -> Dict:
    """
    Verify if extracted value matches reference value using LLM as judge.

    Args:
        client: LLM client for making API calls
        variable_name: Name of the variable being verified
        reference_value: The ground truth value
        extracted_value: The extracted value from model response
        retry_count: Number of retry attempts

    Returns:
        Dict with verification results
    """
    # Handle None/null values with simple logic first
    if reference_value is None and extracted_value is None:
        return {
            "is_correct": True,
            "reasoning": "Both values are null",
            "confidence": "high"
        }
    question_context = question_data["text"]
    prompt = LLM_JUDGE_VERIFICATION_PROMPT.format(
        variable_name=variable_name,
        reference_value=json.dumps(reference_value) if reference_value is not None else "null",
        extracted_value=json.dumps(extracted_value) if extracted_value is not None else "null",
        question_context=question_context or "No additional context provided"
    )

    input_content = prompt

    for attempt in range(retry_count):
        try:
            result = client.chat(input_content, LLMJudgeVerification)
            result["variable_name"] = variable_name
            result["reference_value"] = reference_value
            result["extracted_value"] = extracted_value
            return result
        except Exception as e:
            if attempt < retry_count - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                return {
                    "is_correct": False,
                    "reasoning": f"Error during LLM judge verification: {str(e)}",
                    "confidence": "low",
                    "error": str(e)
                }



def verify_all_answers(reference_values: List, extracted_values: List,
                       variable_list: List[str],
                       client=None, question_data: str = "") -> Dict:
    """
    Verify all extracted values against reference values.

    Args:
        reference_values: List of ground truth values
        extracted_values: List of extracted values
        variable_list: List of variable names
        client: LLM client (required if using llm_judge)
        question_context: Context from the question (for LLM judge)

    Returns:
        Dict with detailed verification results
    """
    if len(reference_values) != len(extracted_values):
        print(reference_values, extracted_values)
        raise InterruptedError

    per_variable_results = []
    all_correct = True
    correct_count = 0

    for var_name, ref_val, ext_val in zip(variable_list, reference_values, extracted_values):
        result = {
            "reference_value": ref_val,
            "extracted_value": ext_val,
        }

        # Perform verification based on method
        verification = verify_answer_with_llm_judge(
            client=client,
            variable_name=var_name,
            reference_value=ref_val,
            extracted_value=ext_val,
            question_data=question_data
        )
        result.update(verification)

        per_variable_results.append(result)

        if result.get("is_correct") == True:
            correct_count += 1
        elif result.get("is_correct") == False:
            all_correct = False

    return {
        "overall_correct": all_correct,
        "variable_correct_count": correct_count,
        "variable_total_count": len(variable_list),
        "variable_accuracy": correct_count / len(variable_list) if len(variable_list) > 0 else 0,
        "per_variable_results": per_variable_results,
    }




def evaluate_sample_passk(judge_client, sample: Dict, k: int, image_folder) -> Dict:
    """
    Evaluate a single sample with pass@k metric.
    Returns dict with generated answers and evaluation results.
    """
    question = sample["question"]
    short_answer_variable = sample["short_answer_variable"]
    short_answer_type = sample["short_answer_type"]
    short_answer_value = sample["short_answer_value"]
    short_answer_description = sample["short_answer_description"]

    # Generate k answers
    generated_answers = sample["generated_answers"]


    # Judge each answer with detailed results
    generated_with_judgments = []

    for i, gen_answer in enumerate(generated_answers):
        extraction_result = extract_values_from_response(
            client=judge_client,
            question=question,
            variable_list=short_answer_variable,
            type_list=short_answer_type,
            reference_description=short_answer_description,
            response=gen_answer,
            id=sample["id"],
        )
        extracted_values = extraction_result["extract_short_answer_value_list"]

        verification_results = verify_all_answers(
            reference_values=short_answer_value,
            extracted_values=extracted_values,
            variable_list=short_answer_variable,
            client=judge_client,
            question_data=question
        )

        # Create detailed entry for each generated answer
        generated_with_judgments.append({
            "answer_index": i + 1,
            "generated_answer": gen_answer,
            "extracted_values": extracted_values,
            "overall_correct": verification_results["overall_correct"],
            "variable_correct_count": verification_results["variable_correct_count"],
            "variable_total_count": verification_results["variable_total_count"],
            "variable_accuracy": verification_results["variable_accuracy"],
            "per_variable_results": verification_results["per_variable_results"],
        })

        time.sleep(0.1)  # Small delay between judge calls

    # Calculate pass@k: at least one correct answer
    num_correct = sum(1 for j in generated_with_judgments if j["overall_correct"])
    n = len(generated_answers)

    # Calculate pass@k using the unbiased estimator
    passk_score = pass_at_k(n, num_correct, k)

    avg_variable_accuracy = sum(j["variable_accuracy"] for j in generated_with_judgments) / len(generated_with_judgments) if generated_with_judgments else 0.0

    # Calculate individual pass rate
    individual_pass_rate = num_correct / k if k > 0 else 0.0

    new_information = {
        "generated_answers_with_judgments": generated_with_judgments,
        "pass_at_k_metrics": {
            "k": k,
            "n": n,
            "num_correct": num_correct,
            "num_incorrect": k - num_correct,
            "passk_unbiased": passk_score,  # Proper pass@k metric
            "individual_pass_rate": individual_pass_rate,
        },

        # Legacy fields for backwards compatibility
        "generated_answers": generated_answers,
        "num_correct": num_correct,
        "avg_variable_accuracy": avg_variable_accuracy,
        "passk_unbiased": passk_score,  # Proper pass@k metric
        "question_accuracy": individual_pass_rate,

    }
    sample["judgment_llm"] = new_information
    return sample


def calculate_passk_metrics(results: List[Dict], k: int) -> Dict:
    """Calculate overall pass@k metrics from evaluation results."""
    total_samples = len(results)
    pass_k_sample = sum(r['judgment_llm']["passk_unbiased"] for r in results)

    passk = pass_k_sample / total_samples if total_samples > 0 else 0.0

    overall_avg_variable_accuracy = sum(r['judgment_llm']["avg_variable_accuracy"] for r in results) / total_samples if total_samples > 0 else 0.0
    overall_question_accuracy = sum(
        r['judgment_llm']["question_accuracy"] for r in results) / total_samples if total_samples > 0 else 0.0

    # Classify overall difficulty

    return {
        "pass@k": passk,
        "k": k,
        "total_samples": total_samples,
        "overall_avg_variable_accuracy": overall_avg_variable_accuracy,
        "overall_question_accuracy": overall_question_accuracy,
    }


def process_sample_wrapper(sample, judge_client, k, image_folder=None):
    """Wrapper function for parallel processing."""
    try:
        return evaluate_sample_passk(judge_client, sample, k, image_folder)
    except Exception as e:
        print(f"Error processing sample: {e}")
        return None


def save_json(filepath: str, data):
    """Save data to JSON file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved results to {filepath}")
    except Exception as e:
        print(f"Error saving to {filepath}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate pass@k using API")
    parser.add_argument('--input', type=str, default='CFE_text_k_1_test_gemini-3-flash-preview.json',
                        help='Input JSON file path')
    parser.add_argument('--input_benchmark', type=str, default='CFE_text.json',
                        help='Input JSON file path')
    parser.add_argument('--image_folder', type=str, default='./',
                        help='MM data image folder path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path')
    parser.add_argument('--k', type=int, default=1,
                        help='Number of answers to generate per question (k in pass@k)')
    parser.add_argument("--judge_provider", default="openai")
    parser.add_argument("--judge_model", default="gpt-5-mini-2025-08-07")
    parser.add_argument("--judge_api_key", dest="judge_api_key",
                        default="Your API Here")
    parser.add_argument('--max_workers', type=int, default=2,
                        help='Number of parallel workers (be careful with rate limits)')

    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        data_samples = json.load(f)

    print(f"Loading bench from {args.input_benchmark}...")
    with open(args.input_benchmark, 'r', encoding='utf-8') as f:
        benchmark_data = json.load(f)

    # check input file
    data_ids = {}
    for i in range(len(data_samples)):
        if len(data_samples[i]["generated_answers"]) != args.k:
            print("The number of generated answers does not match the args.k provided.")
            raise InterruptedError
        data_ids[data_samples[i]["id"]] = data_samples[i]["generated_answers"]
    for i in range(len(benchmark_data)):
        if benchmark_data[i]["id"] not in data_ids:
            id = benchmark_data[i]["id"]
            print(f"id: {id} in {args.input_benchmark} was not found in {args.input}")
            raise InterruptedError

        benchmark_data[i]["generated_answers"] = data_ids[benchmark_data[i]["id"]]

    if not args.output:
        judge_model_name = args.judge_model.replace('/', '-').replace('_', '-')
        output_file_basic = args.input.replace('.json', f'_k_{args.k}_test_{judge_model_name}.json')
    else:
        output_file_basic = args.output

    print(args)
    judge_model_config = {
        "provider": args.judge_provider,
        "model": args.judge_model,
        "api_key": args.judge_api_key,
    }



    data_samples_filtered = benchmark_data
    judge_client = JudgeClient(judge_model_config)

    results = []
    failed_samples = []
    if args.max_workers > 1:
        print("Using parallel processing...")
        process_func = partial(
            process_sample_wrapper,
            judge_client = judge_client,
            k=args.k,
            image_folder=args.image_folder,
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(process_func, sample) for sample in data_samples_filtered]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                result = future.result()
                if result:
                    results.append(result)
                else:
                    failed_samples.append(None)  # Track failures
    else:
        print("Using sequential processing...")
        for sample in tqdm(data_samples_filtered):
            result = evaluate_sample_passk(judge_client = judge_client, sample = sample, k = args.k, image_folder=args.image_folder,)
            if result:
                results.append(result)

    print(f"\nSuccessful: {len(results)}/{len(data_samples_filtered)}")
    print(f"Failed: {len(failed_samples)}")

    # Calculate metrics
    metrics = calculate_passk_metrics(results, args.k)

    # Print results
    print("\n" + "=" * 50)
    print("PASS@K EVALUATION RESULTS")
    print("=" * 50)
    print(f"Pass@{metrics['k']}: {metrics['pass@k']:.2%}")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"overall_avg_variable_accuracy: {metrics['overall_avg_variable_accuracy']:.2f}")
    print(f"overall_question_accuracy: {metrics['overall_question_accuracy']:.2f}")
    print("=" * 50 + "\n")


    # Save results
    output_data = {
        "metrics": metrics,
        "config": {
            "k": args.k,
            "judge_model": args.judge_model,
            "n_samples": len(data_samples_filtered),
            "input_file": args.input,
            "input_benchmark": args.input_benchmark,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "detailed_results": results
    }

    save_json(output_file_basic, results)
    # Save summary
    summary_file = output_file_basic.replace('.json', '_summary.json')
    save_json(summary_file, {"metrics": metrics, "config": output_data["config"]})






if __name__ == "__main__":
    main()