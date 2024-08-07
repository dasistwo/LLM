import os
import re
import csv
import argparse
import pandas as pd
import numpy as np

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
            if not tokens_per_second:
                print(f"Missing required data in file: {filepath}")
                return None
            return {
                'tokens_per_second': float(tokens_per_second.group(1)),
            }
    except IOError as e:
        print(f"Error reading file {filepath}: {e}")
    except Exception as e:
        print(f"Unexpected error processing file {filepath}: {e}")
    return None

def process_log_directory(log_dir):
    data = []
    for filename in os.listdir(log_dir):
        if filename.endswith('.log'):
            file_info = parse_filename(filename)
            if file_info:
                log_data = extract_data_from_log(os.path.join(log_dir, filename))
                if log_data:
                    row = {**file_info, **log_data}
                    data.append(row)
                    print(f"Successfully processed: {filename}")
                else:
                    print(f"No data extracted from {filename}")
            else:
                print(f"Couldn't parse filename: {filename}")
    return pd.DataFrame(data)

def main(rel_dir, dev_dir, output_file):
    if rel_dir and dev_dir:
        # Process both directories and merge
        df_rel = process_log_directory(rel_dir)
        df_dev = process_log_directory(dev_dir)
        
        # Merge dataframes
        merged_df = pd.merge(df_rel, df_dev, on=['model_id', 'model_size', 'precision', 'batch_size', 'input_sequence', 'output_sequence'], suffixes=('_rel', '_dev'), how='outer')
        
        # Calculate gain
        merged_df['gain'] = np.where(
            (merged_df['tokens_per_second_rel'].notna()) & (merged_df['tokens_per_second_dev'].notna()),
            ((merged_df['tokens_per_second_dev'] - merged_df['tokens_per_second_rel']) / merged_df['tokens_per_second_rel'] * 100).round(2),
            np.nan
        )
        
        # Save merged result
        merged_df.to_csv(output_file, index=False)
        print(f"Merged CSV file '{output_file}' has been created.")
    elif rel_dir or dev_dir:
        # Process single directory
        df = process_log_directory(rel_dir or dev_dir)
        output_file = output_file or 'log_analysis.csv'
        df.to_csv(output_file, index=False)
        print(f"CSV file '{output_file}' has been created.")
    else:
        print("Error: At least one log directory (--rel or --dev) must be provided.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process log files and create a CSV analysis with optional merging.")
    parser.add_argument("-r", "--rel", type=str, help="Directory containing the release (control) log files")
    parser.add_argument("-d", "--dev", type=str, help="Directory containing the develop (experiment) log files")
    parser.add_argument("-o", "--out", type=str, help="Path to the output CSV file")
    
    args = parser.parse_args()
    
    main(args.rel, args.dev, args.out)
