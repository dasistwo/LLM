#!python3

from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM
from awq.evaluation import evaluate_perplexity as Perplexity
import os

model_dir = "/data/storage1/model/huggingface/llama/2-7B"
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
model = AutoAWQForCausalLM.from_pretrained(model_dir)
model.quantize(tokenizer, quant_config=quant_config)

tmpdirname = os.path.join(os.getcwd(), 'test_quantized_awq')
model.save_quantized(tmpdirname)
model = AutoAWQForCausalLM.from_quantized(tmpdirname)

Perplexity(model, tokenizer)
