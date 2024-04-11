#!/bin/bash
if [[ -f /.dockerenv ]]; then
    TRTLLM_EXAMPLE_PATH=/app/tensorrt_llm/examples
    MODEL_PATH=/mnt/model
else
    TRTLLM_EXAMPLE_PATH=/data/storage1/jychoi/TensorRT-LLM/examples
    MODEL_PATH=/data/storage1/model
fi

function gemma_convert(){
  local convert_py=${TRTLLM_EXAMPLE_PATH}/gemma/convert_checkpoint.py
  python3 ${convert_py} --ckpt-type torch --model-dir $1 --dtype $3 --world-size 1 --output-model-dir $2
  return $?
}

function llama_convert(){
  local convert_py=${TRTLLM_EXAMPLE_PATH}/llama/convert_checkpoint.py
  python3 ${convert_py} --meta_ckpt_dir $1 --dtype $3 --tp_size 1 --output_dir $2
  return $?
}

function quantize_convert(){
  local convert_py=${TRTLLM_EXAMPLE_PATH}/quantization/quantize.py
  python3 ${convert_py} --model_dir $1 --dtype bfloat16 --qformat $4 --output_dir $2 --tp_size 1
  return $?
}


declare -a prec=()
while [[ "$1" =~ ^- && ! "$1" == -- ]]; do case $1 in
  -p | --precision )
    shift
    while [[ "$1" != -- && "$1" != -* && -n "$1" ]]; do
      prec+=($1)
      echo "Precision: $1"
      shift
    done
    ;;
  -s | --size )
    shift; MODEL_SIZE=$1
    echo "Model size: $1"
    shift
    ;;
  -m | --model )
    shift; MODEL_NAME=$1
    echo "Model name: $1"
    shift
    ;;
  -a | --all )
    ALL_PREC=1
    shift
    ;;
  -c | --clean )
    # Set the clean option flag
    CLEAN_OPTION=1
    echo "Clean option is set"
    shift
    ;;
  -h | --help )
    echo "Usage: [option] [argument]"
    echo "  -p, --precision  <arg>  Specify the precision of the model"
    echo "  -s, --size       <arg>  Specify the size of the model"
    echo "  -m, --model      <arg>  Specify the name of the model"
    echo "  -a, --all               Run all models"
    echo "  -h, --help              Display help"
    echo "  -c, --clean             Clean the existing model"
    exit
    ;;
  * )
    echo "Invalid option: $1"
    exit
    ;;
esac; done
if [[ "$1" == '--' ]]; then shift; fi

if [[ "${#prec[@]}" == 0 && -z "$ALL_PREC" ]]; then
  echo "${#prec[@]}"
  echo "Precision not specified"
  exit
elif [[ -z "$MODEL_SIZE" ]]; then
  echo "Model size not specified"
  exit
elif [[ -z "$MODEL_NAME" ]]; then
  echo "Model name not specified"
  exit
fi

if [[ -n "$ALL_PREC" ]]; then
  prec=("int8_sq" "int4_awq" "int4_wo" "fp16")
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

  if [[ ${PRECISION:5} == "awq" || ${PRECISION:5} == "sq" ]]; then
    PLUGIN="bfloat16"
    TYPED="--strongly_type"
  elif [[ ${PRECISION:5} == "wo" ]]; then
    PLUGIN="bfloat16"
    TYPED=""
  else
    PLUGIN="float16"
    TYPED=""
  fi

  echo "Model: ${MODEL_NAME} Size: ${MODEL_SIZE} Precision: ${PRECISION}"
  if [[ -n "$CLEAN_OPTION" ]]; then
    echo "------------------------------------"
    echo "|    Removing model and engine     |"
    echo "------------------------------------"
    rm -rf ${UNIFIED_CKPT_PATH}
    rm -rf ${ENGINE_PATH}
  fi

  if [[ ! -d "${UNIFIED_CKPT_PATH}" ]]; then
    echo "------------------------------------"
    echo "|      Converting checkpoints      |"
    echo "------------------------------------"
    if [[ $PRECISION == "fp16" ]]; then
      if [[ ! -d "${CKPT_PATH}" ]]; then
        echo "Checkpoint not found"
        exit
      fi
      case $MODEL_NAME in
        "gemma")
          gemma_convert ${CKPT_PATH} ${UNIFIED_CKPT_PATH} ${PLUGIN}
          ;;
        "llama")
          llama_convert ${CKPT_PATH} ${UNIFIED_CKPT_PATH} ${PLUGIN}
          ;;
        *)
          echo "Model name not found"
          exit
          ;;
      esac
    else
      if [[ ! -d "${HF_MODEL_PATH}" ]]; then
        echo "Huggingface model not found"
        exit
      fi
      quantize_convert ${HF_MODEL_PATH} ${UNIFIED_CKPT_PATH} ${PLUGIN} ${PRECISION}
    fi
  fi

  echo "------------------------------------"
  echo "| Constructing TensorRT-LLM engine |"
  echo "------------------------------------"

  if [[ $? -ne 0 ]]; then
    echo "Conversion failed"
    exit
  fi

  if [[ ! -d "${ENGINE_PATH}" ]]; then
    mkdir -p ${ENGINE_PATH}
    trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH} --gemm_plugin ${PLUGIN} \
    --gpt_attention_plugin ${PLUGIN} --lookup_plugin ${PLUGIN} --max_batch_size 8 \
    --max_input_len 256 ${TYPED} --max_output_len 256 --log_level verbose --profiling_verbosity detailed \
    --gather_all_token_logits --enable_debug_output --enable_xqa enable --context_fmha enable \
    --output_dir ${ENGINE_PATH} > ${ENGINE_PATH}/build.log
  fi

  if [[ $? -ne 0 ]]; then
    echo "Engine construction failed"
    exit
  fi

  echo "------------------------------------"
  echo "|  Evaluating TensorRT-LLM engine  |"
  echo "------------------------------------"

  python3 ${TRTLLM_EXAMPLE_PATH}/summarize.py --test_trt_llm --engine_dir ${ENGINE_PATH} \
  --max_input_length 256 --batch_size 8 --max_ite 5 --eval_ppl --vocab_file ${VOCAB_FILE_PATH} \
  --data_type ${PLUGIN} --debug_mode > ${ENGINE_PATH}/summarize.log

done