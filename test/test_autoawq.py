#!python3

from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM
from awq.evaluation import evaluate_perplexity as Perplexity
import os

for model_spec in ['3-8B', '2-7B', '2-13B']:
    model_dir = "/home/jychoi/model/hf/llama/" + model_spec
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    quant_config = {"zero_point": True, "w_bit": 4, "version": "GEMM"}
    model = AutoAWQForCausalLM.from_pretrained(model_dir)
    model.quantize(tokenizer, quant_config=quant_config)
    
    bits = quant_config['w_bit']
    group_size = quant_config['q_group_size']
    tmpdirname = os.path.join(
        os.getcwd(), f"../models/awq_{bits}bit_{group_size}g/" + model_spec
    )
    model.save_quantized(tmpdirname)
    model = AutoAWQForCausalLM.from_quantized(tmpdirname)
    
    Perplexity(model, tokenizer)
