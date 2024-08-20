#!python3

from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM
from awq.evaluation import evaluate_perplexity as Perplexity
from auto_gptq import AutoGPTQForCausalLM
from auto_gptq import BaseQuantizeConfig
import os

for model_spec in ['3-8B', '2-7B', '2-13B']:
    model_dir = "/home/jychoi/model/hf/llama/" + model_spec
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    weight_bit = 3
    group_size = 32
    quant_config = {"zero_point": True, "w_bit": weight_bit, 'q_group_size': group_size, "version": "GEMM"}
    model = AutoAWQForCausalLM.from_pretrained(model_dir)
    model.quantize(tokenizer, quant_config=quant_config)
    bits = quant_config['w_bit']
    group_size = quant_config['q_group_size']
    model_dir = "home/jychoi/model/awq/llama/" + model_spec
    model.save_quantized(model_dir)
    quant_config = BaseQuantizeConfig(bits=weight_bit, group_size=group_size)
    model = AutoGPTQForCausalLM.from_pretrained(model_dir, quantize_config=quant_config)
    examples=[tokenizer("how can i explain? can you imagine it? trying my best trying to find happiness if i belonged if i had a home anywhere it's in your heartbeat let me be there")]
    model.quantize(examples)
    model_dir = "home/jychoi/model/truenibble/llama/" + model_spec
    model.save_pretrained(model_dir)
    model = AutoGPTQForCausalLM.from_quantized(model_dir)
    Perplexity(model, tokenizer)
