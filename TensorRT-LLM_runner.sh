#!/bin/zsh

# Default values
BATCH_SIZE=8
MAX_INPUT_LEN=256
MAX_OUTPUT_LEN=256
MAX_SEQ_LEN=512
WORLD_SIZE=1
IS_DOCKER=0
PPL_OPTION=""
TRTLLM_VER="rel"

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
    PPL_OPTION="--gather_all_token_logits"
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
    elif [[ "$#" -eq 0 ]] || [[ $1 == -* ]]; then
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
        shift
    else
        echo "Error: -ncu requires 3 arguments (String, Integer, Integer) or no arguments for prompt input."
        exit 1
    fi
    ;;

  -nsys ) 
    shift; NSYS_OPTION=1
    echo "Profile with NSYS profiler."
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

MODEL_VER=$(echo ${MODEL_SIZE} | cut -d'-' -f1)
TRTLLM_VER=$(pushd ~ && python -c "import tensorrt_llm; print(tensorrt_llm.__version__)" | tail -n 1 && popd)
TRTLLM_VER=$(echo ${TRTLLM_VER} | tail -n 2 | head -n 1)
if [[ ${TRTLLM_VER} == *"dev"* ]]; then TRTLLM_VER="dev"; else TRTLLM_VER="rel"; fi

if [[ -f /.dockerenv ]]; then
    TRTLLM_EXAMPLE_PATH=/app/tensorrt_llm/examples
    MODEL_PATH=/mnt/model
		IS_DOCKER=1
		PROFILE_OUTPUT_PATH="/app"
    RUNFILE_INPUT="/app/encode/${MODEL_NAME}${MODEL_VER}_${BATCH_SIZE}x${MAX_INPUT_LEN}.npy"

else
    TRTLLM_EXAMPLE_PATH=/home/jychoi/TensorRT-LLM/examples
    MODEL_PATH=/home/jychoi/model
    PROFILE_OUTPUT_PATH=/home/jychoi
    RUNFILE_INPUT="/home/jychoi/encode/${MODEL_NAME}${MODEL_VER}_${BATCH_SIZE}x${MAX_INPUT_LEN}.npy"
fi

if [[ ${PPL_OPTION} != "" ]]; then
  if [[ ${BATCH_SIZE} != 1 || ${MAX_OUTPUT_LEN} != 1 ]]; then
    echo "It is recommended to set output sequence length and batch size to 1 to calculate perplexity."
    while true; do
      read -p "Do you want to continue? (y/n): " yn
      case $yn in
        [Yy]* )
          echo "Continue"; break;;
        [Nn]* )
          exit ;;
        * )
          echo "Please answer yes or no." ;;
      esac
    done
    BATCH_SIZE=1
    MAX_OUTPUT_LEN=1
  fi
fi

MAX_SEQ_LEN=$(( MAX_INPUT_LEN + MAX_OUTPUT_LEN ))
HF_MODEL_PATH="${MODEL_PATH}/hf/${MODEL_NAME}/${MODEL_SIZE}"

# Maybe tokenizer.json doesn't work for all models

#if [[ -f "${HF_MODEL_PATH}/tokenizer.json" ]]; then
#    VOCAB_FILE_OPTION="--vocab_file"
#    VOCAB_FILE_PATH="${HF_MODEL_PATH}/tokenizer.json"
#else
    TOKENIZER_MODEL=$(find "${HF_MODEL_PATH}" -name "tokenizer.model" -print -quit)
    if [[ -n "${TOKENIZER_MODEL}" ]]; then
        TOKENIZER_DIR=$(dirname "${TOKENIZER_MODEL}")
        VOCAB_FILE_OPTION="--tokenizer_dir"
        VOCAB_FILE_PATH="${TOKENIZER_DIR}"
    else
        echo "Error: Neither tokenizer.json nor tokenizer.model found in ${HF_MODEL_PATH} or its subfolders"
        exit 1
    fi
#fi

function gemma_convert(){
  local convert_py=${TRTLLM_EXAMPLE_PATH}/gemma/convert_checkpoint.py
  python3 ${convert_py} --ckpt-type hf --model-dir $1 --dtype $3 --world-size ${WORLD_SIZE} --output-model-dir $2
  return $?
}

function llama_convert(){
  local convert_py=${TRTLLM_EXAMPLE_PATH}/llama/convert_checkpoint.py
  python3 ${convert_py} --model_dir $1 --dtype $3 --tp_size ${WORLD_SIZE} --output_dir $2
  return $?
}

function quantize_convert(){
  local convert_py=${TRTLLM_EXAMPLE_PATH}/quantization/quantize.py
  python3 ${convert_py} --model_dir $1 --dtype float16 --qformat $4 --output_dir $2 --tp_size ${WORLD_SIZE}
  return $?
}

KV_FRACTION=0.99

function calculate_gpu_mem_usage(){
  ####################################################
  ####        Calculate free memory of GPU        ####
  ####################################################

  local config_json=${ENGINE_PATH}/config.json
  
  local memory_sizes=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  local num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
  # Initialize the sum variable
  local total_memory=0

  # Iterate over each memory size and sum them
  while IFS= read -r memory; do
      total_memory=$((total_memory + memory))
  done <<< "$memory_sizes"

  # divide it with world_size
  local memory_per_gpu=$((total_memory * WORLD_SIZE / num_gpus))
  echo "Memory per GPU: ${memory_per_gpu} MiB"

  ####################################################
  ####   Calculate the weight size of the model   ####
  ####################################################

  # Extract required parameters using jq
  local vocab_size=$(jq '.pretrained_config.vocab_size' $config_json)
  local hidden_size=$(jq '.pretrained_config.hidden_size' $config_json)
  local num_hidden_layers=$(jq '.pretrained_config.num_hidden_layers' $config_json)
  local num_key_value_heads=$(jq '.pretrained_config.num_key_value_heads' $config_json)
  local num_attention_heads=$(jq '.pretrained_config.num_attention_heads' $config_json)
  local head_size=$(jq '.pretrained_config.head_size' $config_json)
  local intermediate_size=$(jq '.pretrained_config.intermediate_size' $config_json)

  # Extract quantization parameters
  local quant_algo=$(jq -r '.pretrained_config.quantization.quant_algo' $config_json)
  local group_size=$(jq '.pretrained_config.quantization.group_size' $config_json)
  local has_zero_point=$(jq '.pretrained_config.quantization.has_zero_point' $config_json)

  # Extract weight bits from quant_algo
  local weight_bits=$(echo $quant_algo | sed -n 's/W\([0-9]*\)A.*/\1/p')

  # Calculate sizes
  local embedding_size=$((vocab_size * hidden_size * 2))  # Embedding uses full precision

  if [[ has_zero_point == "true" ]]; then
    group_size=$((group_size * 2))
  fi

  local attention_size=$((2 * hidden_size * ( num_key_value_heads + num_attention_heads ) * head_size * weight_bits / 8))
  local attention_scale=$((attention_size / group_size * 2))
  local attention_total=$((attention_size + attention_scale))

  # Assume we have three ffn matrices
  local ffn_size=$((3 * hidden_size * intermediate_size * weight_bits / 8))
  local ffn_scale=$((ffn_size / group_size * 2))
  local ffn_total=$((ffn_size + $ffn_scale))

  local layer_norm_size=$((hidden_size * 2))  # Layer norm uses full precision

  local layer_total=$((attention_total + ffn_total + layer_norm_size))
  local all_layers_total=$((layer_total * num_hidden_layers))

  local lm_head_size=$((hidden_size * vocab_size * 2))  # LM head uses full precision

  local total_weight_size=$((embedding_size + all_layers_total + lm_head_size))
  total_weight_size_mib=$(echo "scale=2; ${total_weight_size} / (1024 * 1024)" | bc)
  echo "Total weight size: ${total_weight_size_mib} MiB"

  ####################################################
  ####    Calculate KV cache size of the model    ####
  ####################################################

  # Assume the maximum kv_cache_size, which doesn't use paged attention and kv cache quantization.
  local kv_cache_size=$((BATCH_SIZE * MAX_SEQ_LEN * num_hidden_layers * num_key_value_heads * head_size * 2))
  kv_cache_size_mib=$(echo "scale=2; ${kv_cache_size} / (1024 * 1024)" | bc)
  echo "KV Cache size: ${kv_cache_size_mib} MiB"

  ####################################################
  ####      Calculate the activation of TRT       ####
  ####################################################
  # hidden_states before qkv_gemm = MAX_BS x MAX_IN_SEQ x HIDDEN 
  # qkv gemm output = MAX_BS x MAX_INP_LEN x HIDDEN x 3 (QKV)
  # mlp 1st FC out = MAX_BS x MAX_INP_LEN x inter size
  # mlp activation out (assume not fused in plugin mode) = 1st FC out

  local activation_size=$(((BATCH_SIZE * MAX_SEQ_LEN * hidden_size * 4 + BATCH_SIZE * MAX_SEQ_LEN * intermediate_size * 2) * 2))
  activation_size_mib=$(echo "scale=2; ${activation_size} / (1024 * 1024)" | bc)
  echo "Activation size: ${activation_size_mib} MiB"

  ####################################################
  ####       Calculate the logit tensor size      ####
  ####################################################
	# Context logit = input_len*vocab_size*float32*batch_size
	# Generation logit = output_len*vocab_size*float32*batch_size
	#
  local logit_size=$((BATCH_SIZE * MAX_SEQ_LEN * vocab_size * 16))
  logit_size_mib=$(echo "scale=2; ${logit_size} / (1024 * 1024)" | bc)
  echo "Logit size: ${logit_size_mib} MiB"

  result=$(echo "${total_weight_size_mib} + ${kv_cache_size_mib} + ${activation_size_mib}" | bc)
  if [[ ${PPL_OPTION} != "" ]]; then
    result=$(echo "${result} + ${logit_size_mib}" | bc)
  fi

  if (( $(echo "${result} < ${memory_per_gpu}" | bc -l) )); then
    KV_FRACTION=0.9
  else
    if [[ ${PPL_OPTION} != "" ]]; then
      KV_FRACTION=$(echo "scale=2; (${memory_per_gpu} - ${total_weight_size_mib} - ${activation_size_mib} - ${logit_size_mib}) / ${kv_cache_size_mib}" | bc -l)
    else
      KV_FRACTION=$(echo "scale=2; (${memory_per_gpu} - ${total_weight_size_mib} - ${activation_size_mib}) / ${kv_cache_size_mib}" | bc -l)
    fi
    KV_FRACTION=$(printf "0.%02d" ${KV_FRACTION##*.})
  fi

  echo "KV Cache free memory fraction: ${KV_FRACTION}"

}

for PRECISION in "${prec[@]}"; do
  UNIFIED_CKPT_PATH="${MODEL_PATH}/trt_llm/${MODEL_NAME}/${MODEL_SIZE}/${PRECISION}/tp${WORLD_SIZE}"
  if [[ -f /.dockerenv ]]; then
      ENGINE_PATH="${TRTLLM_EXAMPLE_PATH}/${MODEL_NAME}/engine/${MODEL_SIZE}/${PRECISION}/tp${WORLD_SIZE}"
  else
      ENGINE_PATH="${MODEL_PATH}/trt_llm_engine/${MODEL_NAME}/${MODEL_SIZE}/${PRECISION}/tp${WORLD_SIZE}"
  fi

  if [[ ${PRECISION:5} == "awq" || ${PRECISION:5} == "sq" ]]; then
    PLUGIN="float16"
  elif [[ ${PRECISION:5} == "wo" ]]; then
    PLUGIN="float16"
  else
    PLUGIN="float16"
  fi

  if [[ -f ${HF_MODEL_PATH}/config.json ]]; then
    NUM_HIDDEN_LAYERS=$(jq -r '.num_hidden_layers' ${HF_MODEL_PATH}/config.json)
    echo "Number of hidden layers: ${NUM_HIDDEN_LAYERS}"
  else
    echo "${HF_MODEL_PATH}/config.json not found"
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
    mkdir -p ${UNIFIED_CKPT_PATH}
    echo "------------------------------------"
    echo "|      Converting checkpoints      |"
    echo "------------------------------------"
    if [[ $PRECISION == "fp16" ]]; then
      case $MODEL_NAME in
        "gemma")
          gemma_convert ${HF_MODEL_PATH} ${UNIFIED_CKPT_PATH} ${PLUGIN}
          ;;
        "llama")
          llama_convert ${HF_MODEL_PATH} ${UNIFIED_CKPT_PATH} ${PLUGIN}
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

  if [[ -f ${ENGINE_PATH}/config.json ]]; then
    max_batch_size=$(jq -r '.build_config.max_batch_size' ${ENGINE_PATH}/config.json)
    if (( max_batch_size < ${BATCH_SIZE} )); then
      echo "Error: max_batch_size is smaller than BATCH_SIZE"
      while true; do
        read -p "Do you want to construct the engine? (y/n): " yn
        case $yn in
          [Yy]* )
            echo "Continue"; rm ${ENGINE_PATH}/*.engine; break;;
          [Nn]* )
            exit ;;
          * )
            echo "Please answer yes or no." ;;
        esac
      done
    fi
  fi

  if [[ -z $(find "${ENGINE_PATH}" -maxdepth 1 -name "*.engine") ]]; then
    trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH} --gemm_plugin ${PLUGIN} \
    --gpt_attention_plugin ${PLUGIN} --lookup_plugin ${PLUGIN} --max_batch_size ${BATCH_SIZE} \
    --max_input_len ${MAX_INPUT_LEN} --max_seq_len ${MAX_SEQ_LEN} --log_level verbose \
    --profiling_verbosity detailed --use_paged_context_fmha enable \
    ${PPL_OPTION} --enable_debug_output --enable_xqa enable --context_fmha enable \
    --output_dir ${ENGINE_PATH} > ${ENGINE_PATH}/build.log
  fi

  if [[ $? -ne 0 ]]; then
    echo "Engine construction failed"
    exit
  fi

  if [[ ! -f ${RUNFILE_INPUT} ]]; then
    echo "------------------------------------"
    echo "|     Generating input data...     |"
    echo "------------------------------------"
    SCRIPT_PATH=$(dirname "$0")
    RUNFILE_PATH=$(dirname ${RUNFILE_INPUT})
    python3 ${SCRIPT_PATH}/packages/tokenizer.py --model_id ${HF_MODEL_PATH} --length ${MAX_INPUT_LEN} \
    --batch_size ${BATCH_SIZE} --output_path ${RUNFILE_PATH}
  fi

  echo "Calculating gpu memory usage..."
  calculate_gpu_mem_usage
  echo "------------------------------------"

  if [[ -n "$NSYS_OPTION" ]]; then
    echo "------------------------------------"
    echo "|  Profiling with NSYS profiler    |"
    echo "------------------------------------"
    nsys profile -t cuda,nvtx,cublas-verbose,cudnn,cusparse-verbose -b fp \
    --output ${PROFILE_OUTPUT_PATH}/nsys-rep/${MODEL_NAME}_${MODEL_SIZE}_${PRECISION}_batch${BATCH_SIZE}_${MAX_INPUT_LEN}x${MAX_OUTPUT_LEN}_${TRTLLM_VER} -f true \
    python3 ${TRTLLM_EXAMPLE_PATH}/run.py --engine_dir ${ENGINE_PATH} --max_input_length ${MAX_INPUT_LEN} --max_output_len ${MAX_OUTPUT_LEN} \
    --input_file ${RUNFILE_INPUT} ${VOCAB_FILE_OPTION} ${VOCAB_FILE_PATH} \
    --kv_cache_free_gpu_memory_fraction ${KV_FRACTION} --kv_cache_enable_block_reuse 
  elif [[ -n "$NCU_OPTION" ]]; then
    echo "------------------------------------"
    echo "|  Profiling with NCU profiler     |"
    echo "------------------------------------"
    
  	echo -e "\nPlease check the NSYS report before enter the kernel name and launch count."
		echo "The kernel name is ${NCU_KERNEL}"
    echo "The launch count is ${NCU_LAUNCH_COUNT}"
    
    #if [[ NCU_SKIP_COUNT -ge $((NUM_HIDDEN_LAYERS*5)) ]]; then
      SUFFIX="gen"
    #else
    #  SUFFIX="sum"
    #fi
    ncu --set full --nvtx -k ${NCU_KERNEL} -s ${NCU_SKIP_COUNT} -c ${NCU_LAUNCH_COUNT} --target-processes all -f \
    -o ${PROFILE_OUTPUT_PATH}/ncu-rep/${MODEL_NAME}_${MODEL_SIZE}_${PRECISION}_batch${BATCH_SIZE}_${MAX_INPUT_LEN}x${MAX_OUTPUT_LEN}_${SUFFIX}_${CONDA_PREFIX##*/}_${TRTLLM_VER} \
    python3 ${TRTLLM_EXAMPLE_PATH}/run.py --engine_dir ${ENGINE_PATH} --max_input_length ${MAX_INPUT_LEN} --max_output_len ${MAX_OUTPUT_LEN} \
    --input_file ${RUNFILE_INPUT} ${VOCAB_FILE_OPTION} ${VOCAB_FILE_PATH} \
    --kv_cache_free_gpu_memory_fraction ${KV_FRACTION}  --kv_cache_enable_block_reuse 
	else 
    echo "------------------------------------"
    echo "|  Evaluating TensorRT-LLM engine  |"
    echo "------------------------------------"
    if [[ ${PPL_OPTION} != "" ]]; then
        PPL_OPTION="--eval_ppl"
        if [[ ! -d "${PROFILE_OUTPUT_PATH}/ppl_log/${TRTLLM_VER}" ]]; then
          mkdir -p ${PROFILE_OUTPUT_PATH}/ppl_log/${TRTLLM_VER}
        fi

        python3 ${TRTLLM_EXAMPLE_PATH}/summarize.py --test_trt_llm --engine_dir ${ENGINE_PATH} \
        --max_input_length ${MAX_INPUT_LEN} --output_len ${MAX_OUTPUT_LEN} --batch_size ${BATCH_SIZE} --max_ite 5 ${PPL_OPTION} \
        ${VOCAB_FILE_OPTION} ${VOCAB_FILE_PATH} --kv_cache_free_gpu_memory_fraction ${KV_FRACTION} \
        --data_type ${PLUGIN} > ${PROFILE_OUTPUT_PATH}/ppl_log/${TRTLLM_VER}/${MODEL_NAME}_${MODEL_SIZE}_${PRECISION}_batch${BATCH_SIZE}_${MAX_INPUT_LEN}x${MAX_OUTPUT_LEN}.log
    
    else
        if [[ ! -d "${PROFILE_OUTPUT_PATH}/tokpersec_log/${TRTLLM_VER}" ]]; then
          mkdir -p ${PROFILE_OUTPUT_PATH}/tokpersec_log/${TRTLLM_VER}
        fi

        python3 ${TRTLLM_EXAMPLE_PATH}/summarize.py --test_trt_llm --engine_dir ${ENGINE_PATH} \
        --max_input_length ${MAX_INPUT_LEN} --output_len ${MAX_OUTPUT_LEN} --batch_size ${BATCH_SIZE} --max_ite 5 ${PPL_OPTION} \
        ${VOCAB_FILE_OPTION} ${VOCAB_FILE_PATH} --kv_cache_free_gpu_memory_fraction ${KV_FRACTION} \
        --data_type ${PLUGIN} > ${PROFILE_OUTPUT_PATH}/tokpersec_log/${TRTLLM_VER}/${MODEL_NAME}_${MODEL_SIZE}_${PRECISION}_batch${BATCH_SIZE}_${MAX_INPUT_LEN}x${MAX_OUTPUT_LEN}.log
    fi

  fi

done
