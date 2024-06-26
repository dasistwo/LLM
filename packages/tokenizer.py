from transformers import AutoTokenizer
import torch
import numpy as np
import argparse

# Argument parser
parser = argparse.ArgumentParser(description="Tokenize text")
parser.add_argument(
    "--text", type=str, help="Text to tokenize"
)
parser.add_argument(
    "--model_id",
    type=str,
    help="Model ID to use for tokenization",
    default="/data/storage1/model/huggingface/gemma/7b",
)
parser.add_argument(
    "--length", type=int, default=256, help="Minimum length of the tokenized text"
)
parser.add_argument(
    "--batch_size", type=int, default=1, help="Batch size for tokenization"
)
args = parser.parse_args()

output = np.zeros((args.batch_size, args.length), dtype=np.int32)

# get the model from the user input or command line argument
model_id = args.model_id if args.model_id else input("Enter the model name: ")
tokenizer = AutoTokenizer.from_pretrained(args.model_id)

if args.text:
    text = open(args.text, 'r').readlines()
    if len(text) < args.batch_size:
        raise ValueError(f"Text file is too short: {len(text)}")
    for i in range(args.batch_size):
        encoded = tokenizer.encode(
            text[i].strip(), max_length=args.length, truncation=True
        )
        if len(encoded) < args.length:
            raise ValueError(f"Tokenized text is too short in line {i}: {len(encoded)}")
        else:
            output[i, :] = encoded
else:
    # use the fineweb BT-10 dataset
    import datasets
    import os
    
    dataset_path = "/data/storage1/model/data/fineweb/sample/10BT"
    if not os.path.exists(dataset_path):
        # Check whether if to download the dataset
        download = input("Download the dataset? (y/n): ")
        if download.lower() == "y":
            ds = datasets.load_dataset("HuggingFaceFW/fineweb",
                                  name="sample-10BT",
                                  split="train",
                                  cache_dir=dataset_path)
        else:
            raise ValueError("Dataset not found")
    else:
        ds = datasets.load_dataset(dataset_path,
                                   split="train")
    
    ds = ds.filter(lambda x: x['language_score'] > 0.9)
    ds = ds.filter(lambda x: x['token_count'] > args.length * 2) # filtering with safety margin
    ds = ds.shuffle()
    
    # extract the ds_npy and tokenize
    for i in range(args.batch_size):
        encoded = tokenizer.encode(
            ds['text'][i], max_length=args.length, truncation=True
        )
        output[i, :] = encoded

np.save(f"encode{args.batch_size}x{args.length}.npy", output)
