from transformers import AutoTokenizer, AutoConfig
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import torch
import argparse
import numpy as np
from huggingface_hub import login

# Argument parser
parser = argparse.ArgumentParser(description="Quantize model")
parser.add_argument(
  "--model_id", type=str, help="Model ID to quantize"
)
parser.add_argument(
    "--bits", type=int, default=4, help="Number of bits for quantization"
)
parser.add_argument(
    "--model_seqlen", type=int, default=4096, help="Sequence length of the model"
)
parser.add_argument(
    "--group_size", type=int, default=-1, help="Group size for quantization"
)
parser.add_argument(
    "--dataset", type=str, default="wikitext2", help="Dataset for quantization"
)
parser.add_argument(
    "--desc_act",
    type=bool,
    default=False,
    help="Whether to use desc_act for quantization",
)
args = parser.parse_args()

# get the model from the user input or command line argument
model_id = args.model_id if args.model_id else input("Enter the model name: ")

# This is the private token. Be careful not to share it with anyone.
tok = "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"
login(tok)

# some Models cannot handle 4096 sequence length, check the model's config.
config = AutoConfig.from_pretrained(model_id)
if config.max_position_embeddings < args.model_seqlen:
    args.model_seqlen = config.max_position_embeddings
    # Warning message
    print(
        f"Warning: The model's max_position_embeddings is {config.max_position_embeddings}.\n"
        f"Setting model_seqlen to {config.max_position_embeddings}..."
    )

# AutoGPTQ doesn't have a automated pipeline for choose datasets.
# github.com/AutoGPTQ/AutoGPTQ/blob/main/examples/quantization/basic_usage_wikitext2.py
def get_wikitext2(seqlen, model):
    from datasets import load_dataset

    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    import random

    random.seed(4409)
    np.random.seed(0)
    torch.random.manual_seed(0)

    traindataset = []
    for _ in range(trainenc.numel()//seqlen):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
    return traindataset, testenc, tokenizer



"""
Bits: The bit size of the quantized model.

Sequence Length: The length of the dataset sequences used for quantization.
Ideally this is the same as the model sequence length.
For some very long sequence models (16+K), a lower sequence length may have to be used.
Note that a lower sequence length does not limit the sequence length of the quantized model.
It only impacts the quantization accuracy on longer inference sequences.

Group_size: Higher numbers use less VRAM, but have lower quantization accuracy. "None" is the lowest possible value.

GPTQ dataset: The dataset used for quantization.
Using a dataset more appropriate to the model's training can improve quantization accuracy.
Note that the GPTQ dataset is not the same as the dataset used to train the model - please refer to the original model repo for details of the training dataset(s).

Act Order: Also known as desc_act.
True results in better quantization accuracy.
Some GPTQ clients have had issues with models that use Act Order plus Group Size, but this is generally resolved now.
"""

traindataset, trainenc, tokenizer = get_wikitext2(args.model_seqlen, model_id)
quantization_config = BaseQuantizeConfig(
	bits=args.bits,
	group_size=args.group_size,
	desc_act=args.desc_act,
)

quant_model = AutoGPTQForCausalLM.from_pretrained(
    model_id,
    quantize_config=quantization_config,
    torch_dtype=torch.float16,
)

quant_model.quantize(traindataset)

# This is the private token. Be careful not to share it with anyone.
tok = "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"
login(tok)

if quantization_config.desc_act is False:
    branch = "main"
else:
    if quantization_config.group_size == 64:
        branch = "gptq-4bit-64g-actorder_True"
    elif quantization_config.group_size == 32:
        branch = "gptq-4bit-32g-actorder_True"
    else:
        branch = "gptq-4bit-128g-actorder_True"

my_model_id = model_id.split("/")[-1]

quant_model.push_to_hub(f"{my_model_id}-GPTQ", revision=branch)
tokenizer.push_to_hub(f"{my_model_id}-GPTQ", revision=branch)
