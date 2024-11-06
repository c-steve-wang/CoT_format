import datasets
import json

def clean_answer(answer):
    """
    Extract only the final numerical answer from the solution.
    The answer typically comes after '####' and contains just the number.
    """
    parts = answer.split('####')
    if len(parts) == 2:
        return parts[1].strip()
    return answer.strip()

def convert_gsm8k_to_alpaca(dataset):
    """
    Converts GSM8K dataset from Huggingface format to Alpaca format.
    Each entry will have:
    - instruction: the question
    - input: empty string
    - output: just the final numerical answer
    """
    alpaca_data = []
    
    for example in dataset:
        alpaca_entry = {
            "instruction": "Please provide only the final numerical answer for the following questions: " + example['question'],
            "input": "",
            "output": clean_answer(example['answer'])
        }
        alpaca_data.append(alpaca_entry)
    
    return alpaca_data

def main():
    # Load dataset from Huggingface
    dataset = datasets.load_dataset("openai/gsm8k", 'main')
    
    # Convert train split
    train_alpaca = convert_gsm8k_to_alpaca(dataset['train'])
    
    # Convert test split
    test_alpaca = convert_gsm8k_to_alpaca(dataset['test'])
    
    # Save converted data
    with open("gsm8k_train_alpaca.json", 'w', encoding='utf-8') as f:
        json.dump(train_alpaca, f, indent=2, ensure_ascii=False)
    
    with open("gsm8k_test_alpaca.json", 'w', encoding='utf-8') as f:
        json.dump(test_alpaca, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully converted {len(train_alpaca)} training examples")
    print(f"Successfully converted {len(test_alpaca)} test examples")

if __name__ == "__main__":
    main()
