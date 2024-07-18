#!python3

from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
from auto_gptq.quantization import BaseQuantizeConfig
from awq.evaluation import evaluate_perplexity as Perplexity
import os

for model_spec in ['2-7B', '2-13B', '3-8B']:
    model_dir = "/data/storage1/model/huggingface/llama/" + model_spec
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    quant_config = BaseQuantizeConfig(bits=4, group_size=128)
    model = AutoGPTQForCausalLM.from_pretrained(model_dir, quantize_config=quant_config)
    model.quantize()
    
    tmpdirname = os.path.join(os.getcwd(), 'test_onlygptq/' + model_spec)
    model.save_pretrained(tmpdirname)
    model = AutoGPTQForCausalLM.from_quantized(tmpdirname)
    
    Perplexity(model, tokenizer)
