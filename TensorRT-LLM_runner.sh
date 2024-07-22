#!/bin/bash

BATCH_SIZE=8
MAX_INPUT_LEN=256
MAX_OUTPUT_LEN=256
WORLD_SIZE=1
IS_DOCKER=0
PPL_OPTION=""

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
    echo "Model size: ${MODEL_SIZE}"
    shift
    ;;
  -m | --model )
    shift; MODEL_NAME=$1
    echo "Model name: ${MODEL_NAME}"
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
  -b | --batch_size )
    shift; BATCH_SIZE=$1
    echo "Batch size: ${BATCH_SIZE}"
    shift
    ;;
  -i | --max_input_len )
    shift; MAX_INPUT_LEN=$1
    echo "Max input length: ${MAX_INPUT_LEN}"
    shift
    ;;
  -o | --max_output_len )
    shift; MAX_OUTPUT_LEN=$1
    echo "Max output length: ${MAX_OUTPUT_LEN}"
    shift
    ;;
  -w | --world_size | -n | --num_gpus)
    shift; WORLD_SIZE=$1
    echo "World size: ${WORLD_SIZE}"

    shift
    ;;
  -ppl )
    shift; PPL_OPTION="--gather_all_token_logits"
    echo "Building model with Perplexity evaluation."
    shift
    ;;
  -ncu )
    shift; NCU_OPTION=1
    echo "Profile with NCU profiler."
    if [[ "$#" -ge 3 ]] && [[ ! $1 == -* ]] && [[ ! $2 == -* ]] && [[ ! $3 == -* ]]; then
		    NCU_KERNEL=$1
        NCU_SKIP_COUNT=$2
        NCU_LAUNCH_COUNT=$3

        # Check if the second and third arguments are integers
        if ! [[ "$NCU_SKIP_COUNT" =~ ^[0-9]+$ ]] || ! [[ "$NCU_LAUNCH_COUNT" =~ ^[0-9]+$ ]]; then
				    echo "Error: Second and third arguments must be integers."
            exit 1
        fi

        echo "NCU Kernel: ${NCU_KERNEL}"
        echo "NCU Skip Count: ${NCU_SKIP_COUNT}"
        echo "NCU Launch Count: ${NCU_LAUNCH_COUNT}"

        shift 3
    elif [[ "$#" -eq 0 ]]; then
        read -p "Enter NCU Kernel: " NCU_KERNEL
        read -p "Enter NCU Skip Count (Integer): " NCU_SKIP_COUNT
        read -p "Enter NCU Launch Count (Integer): " NCU_LAUNCH_COUNT

        # Check if the input values for Skip Count and Launch Count are integers
        if ! [[ "$NCU_SKIP_COUNT" =~ ^[0-9]+$ ]] || ! [[ "$NCU_LAUNCH_COUNT" =~ ^[0-9]+$ ]]; then
            echo "Error: Skip Count and Launch Count must be integers."
            exit 1
        fi

        echo "NCU Kernel: ${NCU_KERNEL}"
        echo "NCU Skip Count: ${NCU_SKIP_COUNT}"
        echo "NCU Launch Count: ${NCU_LAUNCH_COUNT}"
    else
        echo "Error: -ncu requires 3 arguments (String, Integer, Integer) or no arguments for prompt input."
        exit 1
    fi
		shift
    ;;

  -nsys ) 
    shift; NSYS_OPTION=1
    echo "Profile with NSYS profiler."
    shift
    ;;
  -h | --help )
    echo "Usage: [option] [argument]"
    echo "  -p, --precision      <arg>  Specify the precision of the model"
    echo "  -s, --size           <arg>  Specify the size of the model"
    echo "  -m, --model          <arg>  Specify the name of the model"
    echo "  -a, --all                   Run all models"
    echo "  -h, --help                  Display help"
    echo "  -c, --clean                 Clean the existing model"
    echo "  -b, --batch_size     <arg>  Specify the batch size"
    echo "  -i, --max_input_len  <arg>  Specify the maximum input length"
    echo "  -o, --max_output_len <arg>  Specify the maximum output length"
    echo "  -w, --world_size     <arg>  Specify the number of GPUs"
    echo "  -ppl                        Build model with Perplexity evaluation"
    echo "  -nsys                       Specify the nsys profile option"
    echo "  -ncu                 <arg>  Specify the ncu profile option"
    echo "  Set argument with kernel name, skip count and launch count" 
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

if [[ -f /.dockerenv ]]; then
    TRTLLM_EXAMPLE_PATH=/app/tensorrt_llm/examples
    MODEL_PATH=/mnt/model
		IS_DOCKER=1
		PROFILE_OUTPUT_PATH="/app"
    RUNFILE_INPUT="/app/encode${BATCH_SIZE}x${MAX_INPUT_LEN}.npy"

else
    TRTLLM_EXAMPLE_PATH=/data/storage1/jychoi/TensorRT-LLM/examples
    MODEL_PATH=/data/storage1/model
    PROFILE_OUTPUT_PATH=/data/storage1/jychoi
    RUNFILE_INPUT="/data/storage1/jychoi/encode${BATCH_SIZE}x${MAX_INPUT_LEN}.npy"
fi

function gemma_convert(){
  local convert_py=${TRTLLM_EXAMPLE_PATH}/gemma/convert_checkpoint.py
  python3 ${convert_py} --ckpt-type torch --model-dir $1 --dtype $3 --world-size ${WORLD_SIZE} --output-model-dir $2
  return $?
}

function llama_convert(){
	
  local convert_py=${TRTLLM_EXAMPLE_PATH}/llama/convert_checkpoint.py
  python3 ${convert_py} --meta_ckpt_dir $1 --dtype $3 --tp_size ${WORLD_SIZE} --output_dir $2
  return $?
}

function quantize_convert(){
  local convert_py=${TRTLLM_EXAMPLE_PATH}/quantization/quantize.py
  python3 ${convert_py} --model_dir $1 --dtype bfloat16 --qformat $4 --output_dir $2 --tp_size ${WORLD_SIZE}
  return $?
}

for PRECISION in "${prec[@]}"; do
  CKPT_PATH="${MODEL_PATH}/torch/${MODEL_NAME}/${MODEL_SIZE}"
  UNIFIED_CKPT_PATH="${MODEL_PATH}/trt_llm/${MODEL_NAME}/${MODEL_SIZE}/${PRECISION}/tp${WORLD_SIZE}"
  HF_MODEL_PATH="${MODEL_PATH}/huggingface/${MODEL_NAME}/${MODEL_SIZE}"
  VOCAB_FILE_PATH="${HF_MODEL_PATH}/tokenizer.model"
  if [[ -f /.dockerenv ]]; then
      ENGINE_PATH="${TRTLLM_EXAMPLE_PATH}/${MODEL_NAME}/engine/${MODEL_SIZE}/${PRECISION}/tp${WORLD_SIZE}"
  else
      ENGINE_PATH="${MODEL_PATH}/trt_llm_engine/${MODEL_NAME}/${MODEL_SIZE}/${PRECISION}/tp${WORLD_SIZE}"
  fi

  if [[ -f ${ENGINE_PATH}/config.json ]]; then
    NUM_HIDDEN_LAYERS=$(jq -r '.num_hidden_layers' ${ENGINE_PATH}/config.json)
    echo "Number of hidden layers: ${NUM_HIDDEN_LAYERS}"
  else
    echo "config.json not found"
    exit 1
  fi

  echo "Model: ${MODEL_NAME} Size: ${MODEL_SIZE} Precision: ${PRECISION}"
  if [[ -n "$CLEAN_OPTION" ]]; then
    echo "------------------------------------"
    echo "|         Removing engine          |"
    echo "------------------------------------"
    rm ${ENGINE_PATH}/config.json
    rm ${ENGINE_PATH}/*.engine
  fi

  if [[ $? -ne 0 ]]; then
    echo "No files to remove."
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

  if [[ $? -ne 0 ]]; then
    echo "Conversion failed"
    exit
  fi

  echo "------------------------------------"
  echo "| Constructing TensorRT-LLM engine |"
  echo "------------------------------------"

  if [[ ! -d "${ENGINE_PATH}" ]]; then
    mkdir -p ${ENGINE_PATH}
  fi
  if [[ -z $(find "${ENGINE_PATH}" -maxdepth 1 -name "*.engine") ]]; then
    trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH} --gemm_plugin ${PLUGIN} \
    --gpt_attention_plugin ${PLUGIN} --lookup_plugin ${PLUGIN} --max_batch_size ${BATCH_SIZE} \
    --max_input_len ${MAX_INPUT_LEN} ${TYPED} --max_output_len ${MAX_OUTPUT_LEN} --log_level verbose --profiling_verbosity detailed --use_paged_context_fmha enable \
    ${PPL_OPTION} --enable_debug_output --enable_xqa enable --context_fmha enable \
    --output_dir ${ENGINE_PATH} > ${ENGINE_PATH}/build.log
  fi

  if [[ $? -ne 0 ]]; then
    echo "Engine construction failed"
    exit
  fi

  if [[ ! -f ${RUNFILE_INPUT} ]]; then
    echo "------------------------------------"
    echo "|Generating input data... chill out|"
    echo "------------------------------------"
  fi

  if [[ -n "$NSYS_OPTION" ]]; then
    echo "------------------------------------"
    echo "|  Profiling with NSYS profiler    |"
    echo "------------------------------------"
    nsys profile -t cuda,nvtx,cublas-verbose,cudnn,cusparse-verbose --gpu-metrics-device all -b fp \
    --output ${PROFILE_OUTPUT_PATH}/nsys-rep/${MODEL_NAME}_${MODEL_SIZE}_${PRECISION}_batch${BATCH_SIZE} -f true \
    python3 ${TRTLLM_EXAMPLE_PATH}/run.py --engine_dir ${ENGINE_PATH} --max_output_len ${MAX_OUTPUT_LEN} \
    --input_file ${RUNFILE_INPUT} --vocab_file ${VOCAB_FILE_PATH} \
    --kv_cache_free_gpu_memory_fraction 0.9 --kv_cache_enable_block_reuse 
  elif [[ -n "$NCU_OPTION" ]]; then
    echo "------------------------------------"
    echo "|  Profiling with NCU profiler     |"
    echo "------------------------------------"
    
  	echo -e "\nPlease check the NSYS report before enter the kernel name and launch count."
		echo "The kernel name is ${NCU_KERNEL}"
    echo "The launch count is ${NCU_LAUNCH_COUNT}"
    
    if [[ NCU_SKIP_COUNT -ge $((NUM_HIDDEN_LAYERS*5)) ]]; then
      SUFFIX="gen"
    else
      SUFFIX="sum"
    fi
    ncu --set full --nvtx -k ${NCU_KERNEL} -s ${NCU_SKIP_COUNT} -c ${NCU_LAUNCH_COUNT} --target-processes all -f \
    -o ${PROFILE_OUTPUT_PATH}/ncu-rep/${MODEL_NAME}_MLP_${MODEL_SIZE}_${PRECISION}_batch${BATCH_SIZE}_${SUFFIX} \
    python3 ${TRTLLM_EXAMPLE_PATH}/run.py --engine_dir ${ENGINE_PATH} --max_output_len ${MAX_OUTPUT_LEN} \
    --input_file ${RUNFILE_INPUT} --vocab_file ${VOCAB_FILE_PATH} \
    --kv_cache_free_gpu_memory_fraction 0.9 --kv_cache_enable_block_reuse 
	else 
    echo "------------------------------------"
    echo "|  Evaluating TensorRT-LLM engine  |"
    echo "------------------------------------"
		if [[ ${PPL_OPTION} != "" ]]; then
		  PPL_OPTION="--eval_ppl"
		fi

    python3 ${TRTLLM_EXAMPLE_PATH}/summarize.py --test_trt_llm --engine_dir ${ENGINE_PATH} \
    --max_input_length ${MAX_INPUT_LEN} --batch_size ${BATCH_SIZE} --max_ite 5 ${PPL_OPTION} --vocab_file ${VOCAB_FILE_PATH} --kv_cache_free_gpu_memory_fraction 0.1 \
    --data_type ${PLUGIN} --debug_mode > ${TRTLLM_EXAMPLE_PATH}/${MODEL_NAME}/summarize_${MODEL_SIZE}_${PRECISION}_batch${BATCH_SIZE}.log
  fi

done
