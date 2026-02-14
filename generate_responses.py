import base64
import argparse
import json
import os
from google import genai
from google.genai import types
from typing import Union, List, Dict
from tqdm import tqdm
import time
import concurrent.futures
from functools import partial
import logging

class _NoFunctionCallWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if "there are non-text parts in the response:" in message:
            return False
        else:
            return True
logging.getLogger("google_genai.types").addFilter(_NoFunctionCallWarning())

def encode_image_gemini(image_path):
    with open(image_path, "rb") as image_file:
        return image_file.read()

class LLMClient:
    def __init__(self, model_config):
        api_key = model_config["api_key"]
        self.provider = model_config["provider"]
        self.model_name = model_config["model"]

        if model_config["provider"] == "gemini":
            self.client = genai.Client(api_key=api_key)
        else:
            raise Exception("The provider is not supported")

    def chat(self, input):
        if self.provider == "gemini":
            if "gemma" in self.model_name:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=input,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        top_k=1),
                )
            else:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=input,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        top_k=1,
                        thinking_config=types.ThinkingConfig(thinking_level="high")),
                )
            return response.text
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": input}],
                temperature=self.temperature,
                # top_k=1,
            )
            return response.choices[0].message.content


# Generation prompt template
GENERATION_PROMPT_TEMPLATE = """
Answer the following question step-by-step:

**Question:**
{question}

Provide a direct, accurate answer based on the information available."""



def generate_answer(client, question, image_folder="./", retry_count: int = 2) -> str:
    """Generate a single answer for a question using Gemini."""
    prompt = GENERATION_PROMPT_TEMPLATE.format(question=question["text"])
    if len(question["images"]) == 0:
        input_content = prompt
    else:
        if client.provider == "gemini":
            images_question = [encode_image_gemini(os.path.join(image_folder, image_path_i)) for image_path_i in
                               question["images"]]
            images = images_question

            input_content = [prompt]
            for image in images:
                input_content.append(
                    types.Part.from_bytes(
                    data=image,
                    mime_type='image/jpeg',
                 ))

    backoff = 2
    for i in range(retry_count):
        try:
            pred = client.chat(input_content)
            return pred
        except Exception as exc:
            print(exc)
            if "429" in str(exc) or "503" in str(exc):
                print(f"API busy. Retrying in {backoff}s... (Attempt {i + 1}/{retry_count})")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue
            else:
                print(f"Error generating answer: {exc} ... (Attempt {i + 1}/{retry_count})")
                continue
    return ""


def evaluate_sample_passk(test_client, sample: Dict, k: int, image_folder) -> Dict:
    """
    Generate answer with k
    Returns dict with generated answers and id.
    """
    question = sample["question"]
    id = sample["id"]

    # Generate k answers
    generated_answers = []
    for _ in range(k):
        answer = generate_answer(test_client, question, image_folder=image_folder)
        generated_answers.append(answer)
        time.sleep(0.1)  # Small delay between generations

    save_response = {"generated_answers": generated_answers, "id": id}
    return save_response



def process_sample_wrapper(sample, test_client, k, image_folder="./"):
    """Wrapper function for parallel processing."""
    try:
        return evaluate_sample_passk(test_client, sample, k, image_folder)
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
    parser.add_argument('--input', type=str, default='CFE_text.json',
                        help='Input JSON file path')
    parser.add_argument('--image_folder', type=str, default='./',
                        help='MM data image folder path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path')
    parser.add_argument('--k', type=int, default=1,
                        help='Number of answers to generate per question (k in pass@k)')
    parser.add_argument("--test_provider", default="gemini")
    parser.add_argument("--test_model", default="gemini-3-flash-preview")
    parser.add_argument("--test_api_key", dest="test_api_key",
                        default="Your API Here")
    parser.add_argument('--max_workers', type=int, default=2,
                        help='Number of parallel workers (be careful with rate limits)')

    args = parser.parse_args()


    if not args.output:
        test_model_name = args.test_model.replace('/', '-').replace('_', '-')
        output_file_basic = args.input.replace('.json', f'_k_{args.k}_test_{test_model_name}.json')
    else:
        output_file_basic = args.output

    print(args)
    print(f"Loading data from {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        data_samples = json.load(f)

    # Initialize models
    test_model_config = {
        "provider": args.test_provider,
        "model": args.test_model,
        "api_key": args.test_api_key,
    }

    print(f"Testing Model: {args.test_model}")
    test_client = LLMClient(test_model_config)

    results = []
    failed_samples = []
    if args.max_workers > 1:
        print("Using parallel processing...")
        process_func = partial(
            process_sample_wrapper,
            test_client = test_client,
            k=args.k,
            image_folder=args.image_folder,
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(process_func, sample) for sample in data_samples]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                result = future.result()
                if result:
                    results.append(result)
                else:
                    failed_samples.append(None)  # Track failures
    else:
        print("Using sequential processing...")
        for sample in tqdm(data_samples):
            result = evaluate_sample_passk(test_client = test_client, sample = sample, k = args.k, image_folder=args.image_folder,)
            if result:
                results.append(result)

    print(f"\nSuccessful: {len(results)}/{len(data_samples)}")
    print(f"Failed: {len(failed_samples)}")


    save_json(output_file_basic, results)


if __name__ == "__main__":
    main()