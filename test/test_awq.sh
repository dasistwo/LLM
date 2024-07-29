#!/bin/zsh

for model_spec in "3-8B" "2-7B" "2-13B"; do
	for bits in 3 4; do
		MODEL_PATH="/data/storage1/model/hf/llama/"${MODEL_SPEC}

		# generate real quantized weights (w4)
	  python -m awq.entry --model_path ${MODEL_PATH} \
    	--q_backend real \
    	--w_bit ${bits} --q_group_size 128 \
    	--load_awq ~/quant_libs/llm-awq/awq_cache/llama-${model_spec}-w${bits}-g128.pt \
			--dump_quant quant_cache/${MODEL_PATH}-w${bits}-g128-awq.pt

		# load and evaluate the real quantized model (smaller gpu memory usage)
	  python -m awq.entry --model_path ${MODEL_PATH} \
  	  --tasks wikitext \
  	  --w_bit ${bits} --q_group_size 128 \
			--load_quant quant_cache/${MODEL_PATH}-w${bits}-g128-awq.pt
	done
done
