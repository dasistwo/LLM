#!python3

from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
from auto_gptq import BaseQuantizeConfig
from awq.evaluation import evaluate_perplexity as Perplexity
import os

for model_spec in ['3-8B', '2-7B', '2-13B']:
    model_dir = "/home/jychoi/model/hf/llama/" + model_spec
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    quant_config = BaseQuantizeConfig(bits=4, group_size=-1)
    model = AutoGPTQForCausalLM.from_pretrained(model_dir, quantize_config=quant_config)
    #model.quantize()

    examples=[tokenizer("How can I explain? Can you imagine it? Trying my best Trying to find happiness If I belonged If I had a home anywhere It's in your heartbeat Let me be there")]
    model.quantize(examples)

    bits = quant_config.bits
    group_size = quant_config.group_size
    tmpdirname = os.path.join(
        os.getcwd(), f"../models/gptq_{bits}bit_{group_size}g/" + model_spec
    )
    model.save_pretrained(tmpdirname)
    model = AutoGPTQForCausalLM.from_quantized(tmpdirname)

    Perplexity(model, tokenizer)
