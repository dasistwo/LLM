#!python3

from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
from auto_gptq.quantization import BaseQuantizeConfig
from awq.evaluation import evaluate_perplexity as Perplexity
import os

for model_spec in ['3-8B', '2-7B', '2-13B']:
    model_dir = "/home/jychoi/model/hf/llama/" + model_spec
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    quant_config = BaseQuantizeConfig(bits=3, group_size=16)
    model = AutoGPTQForCausalLM.from_pretrained(model_dir, quantize_config=quant_config)
    model.quantize()

    bits = quant_config.bits
    group_size = quant_config.group_size
    tmpdirname = os.path.join(
        os.getcwd(), f"../models/{bits}bit_{group_size}g/" + model_spec
    )
    model.save_pretrained(tmpdirname)
    model = AutoGPTQForCausalLM.from_quantized(tmpdirname)

    Perplexity(model, tokenizer)
