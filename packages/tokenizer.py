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
    for i in range(args.batch_size):
        text = input("Enter the text to tokenize: ")
        output[i, :] = tokenizer.encode(text, max_length=args.length, truncation=True, padding="max_length")
        if output[i].shape[0] < args.length:
            raise ValueError(f"Tokenized text is too short: {output[i].shape[0]}")

np.save(f"encoded{args.batch_size}x{args.length}.npy", output)
