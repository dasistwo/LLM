#!/bin/sh
if [[ -f /.dockerenv ]]; then
    TRTLLM_EXAMPLE_PATH=/app/tensorrt_llm/examples
    MODEL_PATH=/mnt/model
else
    TRTLLM_EXAMPLE_PATH=/data/storage1/jychoi/TensorRT-LLM/examples
    MODEL_PATH=/data/storage1/model
fi

function gemma_convert(){
  local convert_py=${TRTLLM_EXAMPLE_PATH}/gemma/convert_checkpoint.py
  python3 ${convert_py} --ckpt-type torch --model-dir $1 --dtype float16 --world-size 1 --output-model-dir $2
}

function llama_convert(){
  local convert_py=${TRTLLM_EXAMPLE_PATH}/llama/convert_checkpoint.py
  python3 ${convert_py} --model_dir $1 --dtype float16 --tp_size 1 --output_dir $2
}

function quantize_convert(){
  local convert_py=${TRTLLM_EXAMPLE_PATH}/quantization/quantize.py
  python3 ${convert_py} --model_dir $1 --dtype float16 --qformat $3 --output_dir $2 --tp_size 1
}

while [[ "$1" =~ ^- && ! "$1" == "--" ]]; do case $1 in
  -p | --precision )
    shift; PREC=$1
    exit
    ;;
  -s | --size )
    shift; MODEL_SIZE=$1
    ;;
  -m | --model )
    shift; MODEL_NAME=$1
    ;;
  -a | --all )
    ALL_PREC=1
    ;;
  -h | --help )
    echo "Usage: [option] [argument]"
    echo "  -p, --precision  <arg>  Specify the precision of the model"
    echo "  -s, --size       <arg>  Specify the size of the model"
    echo "  -m, --model      <arg>  Specify the name of the model"
    echo "  -a, --all               Run all models"
    echo "  -h, --help              Display help"
    exit
    ;;
  * )
    echo "Invalid option: $1"
    exit
    ;;
esac; shift; done
if [[ "$1" == '--' ]]; then shift; fi

if [[ -z "$PREC" || -z "$ALL_PREC"]]; then
  echo "Precision not specified"
  exit
elif [[ -z "$MODEL_SIZE" ]]; then
  echo "Model size not specified"
  exit
elif [[ -z "$MODEL_NAME" ]]; then
  echo "Model name not specified"
  exit
fi

declare -a prec
if [[ -n "$PREC" ]]; then
  prec+=($PREC)
elif [[ -n "$ALL_PREC" ]]; then
  prec=("fp8", "int8_sq", "int4_awq", "w4a8_awq", "int8_wo", "int4_wo" "fp16")
fi

for PRECISION in "${prec[@]}"; do
  CKPT_PATH="${MODEL_PATH}/torch/${MODEL_NAME}/${MODEL_SIZE}"
  UNIFIED_CKPT_PATH="${MODEL_PATH}/trt_llm/${MODEL_NAME}/${MODEL_SIZE}/${PRECISION}/tp1"
  HF_MODEL_PATH="${MODEL_PATH}/huggingface/${MODEL_NAME}/${MODEL_SIZE}"
  VOCAB_FILE_PATH="${HF_MODEL_PATH}/tokenizer.model"
  if [[ -f /.dockerenv ]]; then
      ENGINE_PATH="${TRTLLM_EXAMPLE_PATH}/${MODEL_NAME}/engine/${MODEL_SIZE}/${PRECISION}/tp1"
  else
      ENGINE_PATH="${MODEL_PATH}/trt_llm_engine/${MODEL_NAME}/${MODEL_SIZE}/${PRECISION}/tp1"
  fi

  if [[ -z "$(ls -A ${UNIFIED_CKPT_PATH})" ]]; then
    if [[ $PRECISION == "fp16" ]]; then
      case $MODEL_NAME in
        "gemma")
          gemma_convert ${CKPT_PATH} ${UNIFIED_CKPT_PATH}
          ;;
        "llama")
          llama_convert ${CKPT_PATH} ${UNIFIED_CKPT_PATH}
          ;;
        *)
          echo "Model name not found"
          exit
          ;;
      esac
    else
      quantize_convert ${CKPT_PATH} ${UNIFIED_CKPT_PATH} ${PRECISION}
    fi
  fi

  if [[ -z "$(ls -A ${ENGINE_PATH})" ]]; then
    trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH} --gemm_plugin float16 \
    --gpt_attention_plugin float16 --lookup_plugin float16 --max_batch_size 8 \
    --max_input_len 256 --max_output_len 256 --log_level verbose --profiling_verbosity detailed \
    --gather_all_token_logits --enable_debug_output --enable_xqa enable --context_fmha enable \
    --output_dir ${ENGINE_PATH} > ${ENGINE_PATH}/build.log
  fi

  python3 ${TRTLLM_EXAMPLE_PATH}/summarize.py --test_trt_llm --engine_dir ${ENGINE_PATH}  --max_input_length 256 --batch_size 8 --max_ite 5 --eval_ppl --vocab_file ${VOCAB_FILE_PATH}

done



