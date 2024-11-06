import json
from typing import List, Dict
import re
from tqdm import tqdm

def clean_number(text: str) -> str:
    """
    Extract and clean numerical answer from text, handling both numbers and commas
    """
    # Remove commas from numbers and find all numbers in the text
    text = text.replace(',', '')
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return numbers[0] if numbers else text.strip()

def calculate_accuracy(input_file: str, output_file: str):
    """
    Calculate accuracy and save detailed results
    """
    # Load predictions
    print(f"Loading predictions from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    correct = 0
    total = 0
    results = []
    
    print("Calculating accuracy...")
    for item in tqdm(data):
        try:
            # Get ground truth and generated answers
            true_answer = clean_number(item['output'])
            generated_answer = clean_number(item['generated_answer'])
            
            # Compare answers
            is_correct = true_answer == generated_answer
            
            # Store detailed results
            result_item = {
                'instruction': item['instruction'],
                'true_answer': true_answer,
                'generated_answer': generated_answer,
                'is_correct': is_correct
            }
            results.append(result_item)
            
            if is_correct:
                correct += 1
            total += 1
            
        except Exception as e:
            print(f"Error processing item: {str(e)}")
            continue
    
    # Calculate accuracy
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    # Prepare final results
    final_results = {
        'accuracy': accuracy,
        'correct_count': correct,
        'total_count': total,
        'detailed_results': results
    }
    
    # Save results
    print(f"Saving results to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nFinal Accuracy: {accuracy:.2f}%")
    print(f"Correct: {correct}/{total}")

def main():
    # Configuration
    INPUT_FILE = "gsm8k_test_alpaca_with_predictions.json"  # Your predictions file
    OUTPUT_FILE = "accuracy_results.json"
    
    # Calculate and save accuracy
    calculate_accuracy(INPUT_FILE, OUTPUT_FILE)

if __name__ == "__main__":
    main()
