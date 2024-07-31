#!python3
import os
import re
import csv

def parse_filename(filename):
    pattern = r'(\w+)_(\d?-?\d+[B|M])_(int4_awq|fp16|int4_wo|int8_sq)_batch(\d+)_(\d+)x(\d+)\.log'
    match = re.match(pattern, filename)
    if match:
        return {
            'model_id': match.group(1),
            'model_size': match.group(2),
            'precision': match.group(3),
            'batch_size': match.group(4),
            'input_sequence': match.group(5),
            'output_sequence': match.group(6)
        }
    return None

def extract_data_from_log(filepath):
    try:
        with open(filepath, 'r') as file:
            content = file.read()

            tokens_per_second = re.search(r'tokens per second: ([\d.]+)', content)
            rouge1 = re.search(r'rouge1 : ([\d.]+)', content)
            rouge2 = re.search(r'rouge2 : ([\d.]+)', content)
            rougeL = re.search(r'rougeL : ([\d.]+)', content)
            rougeLsum = re.search(r'rougeLsum : ([\d.]+)', content)

            if not all([tokens_per_second, rouge1, rouge2, rougeL, rougeLsum]):
                print(f"Missing required data in file: {filepath}")
                return None

            return {
                'tokens_per_second': tokens_per_second.group(1),
                'rouge1': rouge1.group(1),
                'rouge2': rouge2.group(1),
                'rougeL': rougeL.group(1),
                'rougeLsum': rougeLsum.group(1)
            }
    except IOError as e:
        print(f"Error reading file {filepath}: {e}")
    except Exception as e:
        print(f"Unexpected error processing file {filepath}: {e}")
    return None

def main():
    log_dir = '.'  # Current directory
    output_file = 'log_analysis.csv'

    fieldnames = ['model_id', 'model_size', 'precision', 'batch_size', 'input_sequence', 'output_sequence',
                  'tokens_per_second', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum']

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for filename in os.listdir(log_dir):
            if filename.endswith('.log'):
                file_info = parse_filename(filename)
                if file_info:
                    log_data = extract_data_from_log(os.path.join(log_dir, filename))
                    if log_data:
                        row = {**file_info, **log_data}
                        writer.writerow(row)
                        print(f"Successfully processed: {filename}")
                    else:
                        print(f"No data extracted from {filename}")
                else:
                    print(f"Couldn't parse filename: {filename}")

    print(f"CSV file '{output_file}' has been created.")

if __name__ == "__main__":
    main()
