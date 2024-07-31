#!python3
import numpy as np
import argparse
import os
import re
import subprocess

def find_largest_file(model_id, path):
    pattern = re.compile(rf"{model_id}_(\d+)x(\d+)\.npy")
    largest_input_length = 0
    largest_input_batch = 0
    largest_file = None
    
    for filename in os.listdir(path):
        match = pattern.match(filename)
        if match:
            input_batch, input_length = map(int, match.groups())
            if (input_length > largest_input_length) or (input_length == largest_input_length and input_batch > largest_input_batch):
                largest_input_length = input_length
                largest_input_batch = input_batch
                largest_file = filename

    return largest_file, largest_input_batch, largest_input_length

def ask_and_run_tokenizer(model_id, output_batch, output_length, output_file):
    user_input = input("Do you want to create a new file with appropriate dimensions using tokenizer.py? (yes/no): ").strip().lower()
    if user_input == 'yes':
        script_dir = os.path.dirname(os.path.realpath(__file__))
        command = [
            "python3", "tokenizer.py",
            "--model_id", model_id,
            "--length", str(output_length),
            "--batch_size", str(output_batch),
            "--output_path", output_file
        ]
        subprocess.run(command, cwd=script_dir)
        print(f"Created new file {output_file} with tokenizer.py")
        return True
    else:
        print("Operation aborted by the user.")
        return False

def process_array(model_id, output_batch, output_length, path):
    largest_file, input_batch, input_length = find_largest_file(model_id, path)
    output_file = os.path.join(path, f"{model_id}_{output_batch}x{output_length}.npy")

    if largest_file is None or output_batch > input_batch or output_length > input_length:
        if ask_and_run_tokenizer(model_id, output_batch, output_length, output_file):
            return
        else:
            return

    input_file = os.path.join(path, largest_file)

    try:
        arr = np.load(input_file)
        newarr = arr[:output_batch, :output_length]
        np.save(output_file, newarr)
        print(f"Processed {input_file} to {output_file}")

    except Exception as e:
        print(f"An error occurred while processing the file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process numpy arrays with custom parameters.")
    parser.add_argument("--model_id", "-m", type=str, default="llama3", help="Model ID (default: llama3)")
    parser.add_argument("--output_batch", "-ob", type=int, default=32, help="Output batch size (default: 32)")
    parser.add_argument("--output_length", "-ol", type=int, default=256, help="Output sequence length (default: 256)")
    parser.add_argument("--path", "-p", type=str, default=os.path.expanduser("~/encode"), help="Path to the folder containing the files (default: ~/encode)")

    args = parser.parse_args()

    process_array(args.model_id, args.output_batch, args.output_length, args.path)

