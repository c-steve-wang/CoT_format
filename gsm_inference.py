import json
from typing import List, Dict
import torch
from transformers import pipeline
from tqdm import tqdm

def create_pipeline(model_path: str):
    """
    Create the text generation pipeline with the finetuned model
    """
    print("Setting up pipeline...")
    generator = pipeline(
        "text-generation",
        model=model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return generator

def generate_answer(generator, instruction: str, max_length: int = 128) -> str:
    """
    Generate an answer for a given instruction using the pipeline
    """
    response = generator(
        instruction,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True
    )

    full_text = response[0]['generated_text']
    answer = full_text[len(instruction):].strip()
    return answer

def process_dataset(data: List[Dict], model_path: str, output_path: str):
    """
    Process the entire dataset and append generated answers
    """
    # Create pipeline
    generator = create_pipeline(model_path)

    # Process each example
    print("Generating answers...")
    for item in tqdm(data):
        try:
            instruction = "Please provide only the final numerical answer for the following questions: " + item["instruction"]
            generated_answer = generate_answer(generator, instruction)
            item["generated_answer"] = generated_answer

        except Exception as e:
            print(f"Error processing instruction: {instruction}")
            print(f"Error: {str(e)}")
            item["generated_answer"] = "Error generating answer"

    # Save updated dataset
    print(f"Saving results to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    # Configuration
    MODEL_PATH = "saves/llama3-8b/full/sft_gsm8k"  # Update this path
    INPUT_PATH = "./data/gsm8k_test_alpaca.json"
    OUTPUT_PATH = "./gsm8k_test_alpaca_with_predictions.json"

    # Load input dataset
    print(f"Loading dataset from {INPUT_PATH}")
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Process dataset
    process_dataset(data, MODEL_PATH, OUTPUT_PATH)
    print("Processing complete!")

if __name__ == "__main__":
    main()
