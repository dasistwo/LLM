#!/bin/sh
# This script is not runnable by itself. It is rather a note for running wikitext perplexity task.
accelerate launch -m lm-eval --model hf --model_args pretrained=google/gemma-2b --tasks wikitext