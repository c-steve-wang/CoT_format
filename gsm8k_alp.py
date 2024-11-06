import datasets
import json
import re

def clean_text(text):
    """
    Clean text by removing unusual line terminators and normalizing whitespace
    """
    # Replace any type of line terminator with standard newline
    text = re.sub(r'[\u2028\u2029\r\n]+', '\n', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text

def clean_answer(answer):
    """
    Extract only the final numerical answer from the solution.
    """
    parts = answer.split('####')
    if len(parts) == 2:
        return clean_text(parts[1].strip())
    return clean_text(answer.strip())

def convert_gsm8k_to_alpaca(dataset):
    """
    Converts GSM8K dataset from Huggingface format to Alpaca format.
    """
    alpaca_data = []
    
    for example in dataset:
        alpaca_entry = {
            "instruction": clean_text(example['question']),
            "input": "",
            "output": clean_answer(example['answer'])
        }
        alpaca_data.append(alpaca_entry)
    
    return alpaca_data

def save_json(data, filename):
    """
    Save JSON data with proper line endings
    """
    with open(filename, 'w', encoding='utf-8', newline='\n') as f:
        # Use ensure_ascii=False to preserve Unicode characters
        # Use newline='\n' to ensure consistent line endings
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        # Additional cleaning of the entire JSON string
        json_str = re.sub(r'[\u2028\u2029\r\n]+', '\n', json_str)
        f.write(json_str)

def main():
    try:
        # Load dataset from Huggingface
        dataset = datasets.load_dataset("openai/gsm8k", 'main')
        
        # Convert splits
        train_alpaca = convert_gsm8k_to_alpaca(dataset['train'])
        test_alpaca = convert_gsm8k_to_alpaca(dataset['test'])
        
        # Save converted data with clean line endings
        save_json(train_alpaca, "gsm8k_train_alpaca.json")
        save_json(test_alpaca, "gsm8k_test_alpaca.json")
        
        print(f"Successfully converted {len(train_alpaca)} training examples")
        print(f"Successfully converted {len(test_alpaca)} test examples")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
