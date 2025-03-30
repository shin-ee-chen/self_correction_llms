export CUDA_VISIBLE_DEVICES="0,1,2,3"


MODEL_NAME_OR_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
OUTPUT_DIR=DeepSeek-R1-Distill-Qwen-7B

SPLIT="test"
NUM_TEST_SAMPLE=100
DATA_NAME='math500'

PROMPT_TYPE="deepseek-r1"
# DATA_NAME="aime24,aime25,aimo2,math500_level5"
# TOKENIZERS_PARALLELISM=false \
# python3 -u first_reasoning_generation.py \
#     --model_name_or_path ${MODEL_NAME_OR_PATH} \
#     --data_dir "./data" \
#     --data_name ${DATA_NAME} \
#     --output_dir ${OUTPUT_DIR} \
#     --split ${SPLIT} \
#     --prompt_type ${PROMPT_TYPE} \
#     --max_tokens_per_call 32768 \
#     --num_test_sample ${NUM_TEST_SAMPLE} \
#     --seed 0 \
#     --temperature 0.8 \
#     --top_p 0.95 \
#     --start 0 \
#     --end -1 \
#     --n_sampling 16 \

DATASET_DIR='/home/srajaee/self_correction_llms/outputs/DeepSeek-R1-Distill-Qwen-7B/math500/first_reasonings/test_DeepSeek-R1-Distill-Qwen-7B_seed0_t0.8_len32768_num100s0e-1_first_reasoning_dataset.json'
NUM_TEST_SAMPLE=32
python3 -u answer_sampling.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --dataset_dir ${DATASET_DIR} \
    --data_dir "./data" \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --max_tokens_per_call 32768 \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0.7 \
    --top_p 0.95 \
    --start 0 \
    --end -1 \
    --n_sampling 16 \

# DATASET_DIR='/home/srajaee/self_correction_llms/outputs/DeepSeek-R1-Distill-Qwen-7B/math500/predictions/test_DeepSeek-R1-Distill-Qwen-7B_seed0_t0.7_len32768_num100s0e-1_dataset_predictions.jsonl'
# python3 -u judge_answer.py \
#     --model_name_or_path ${MODEL_NAME_OR_PATH} \
#     --dataset_dir ${DATASET_DIR} \
#     --data_name ${DATA_NAME} \
#     --output_dir ${OUTPUT_DIR} \
#     --split ${SPLIT} \
#     --num_test_sample ${NUM_TEST_SAMPLE} \
#     --seed 0 \
#     --start 0 \
#     --end -1 \


    
