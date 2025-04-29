import os
import time
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch
from utils.data import load_generated_data, save_jsonl
from utils.parser import extract_pred_and_parse
from utils.eval import per_sample_verification
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="aime24", type=str)
    parser.add_argument("--dataset_dir", default="/projects/0/gusr0608/self_correction_llms/outputs/modified_first_reasoning/wrong/aime24/predictions/test_DeepSeek-R1-Distill-Qwen-7B_seed0_t0_len16384_num-1s0e-1_True_dataset_predictions.json", type=str)
    parser.add_argument("--model_name_or_path", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", type=str)
    parser.add_argument("--output_dir", default="/projects/0/gusr0608/self_correction_llms/modified_first_reasoning", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)

    args = parser.parse_args()

    return args


def set_seed(seed: int = 42) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    print(f"Random seed set as {seed}")


def prepare_data(data_path, data_name, args):
    examples = load_generated_data(data_path)

    if args.num_test_sample > 0:
        examples = examples[: args.num_test_sample]

    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # get out_file name
    model_name = args.model_name_or_path.split('/')[-1]
    out_file_prefix = f"{args.split}_{model_name}_seed{args.seed}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    
    generated_dataset_file = f"{output_dir}/{data_name}/judge/{out_file_prefix}_num{args.num_test_sample}s{args.start}e{args.end}_dataset_judge_answer.jsonl"

    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)
    os.makedirs(f"{output_dir}/{data_name}/judge", exist_ok=True)
    return examples, generated_dataset_file


def setup(args):
    # load model
    # available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")

    # infer & eval
    data_paths = args.dataset_dir.split(",")
    data_list = args.data_names.split(",")


    assert len(data_list) == len(data_paths)

    results = []
    for i, d in enumerate(data_list):
        main(d, data_paths[i], args)
     
        
def main(data_name, data_path, args):
        examples, generated_dataset_file = prepare_data(data_path, data_name, args)
        print("Data prepration done!")
        print("=" * 50)
        print("data:", data_name, " , #samples:", len(examples))
        tokenizer =  AutoTokenizer.from_pretrained(args.model_name_or_path)

        samples = []
        for i, example in tqdm(enumerate(examples), total=len(examples)):
            sample = {
                "question": example["question"],
                "gt": example['gt'],
                "prompt": example['prompt'],
                "reasonings": example['reasonings'],
                "answer": example['answer'],
                "sample_idx": example['sample_idx']
            }
            samples.append(sample)


        updated_samples = []
        accs = []
        n_tokens = []
        for i, sample in enumerate(samples):
            scores = []
            preds = []
            sample_idx = 0
            
            sampling_scores = []
            sampling_preds = []
            
            for j in range(len(sample['reasonings'])):
                # reasoning_scores = []
                # reasoning_preds = []
                for k in range(len(sample['answer'][j])):
                    if sample["sample_idx"][j] != sample_idx:
                        scores.append(sampling_scores)
                        preds.append(sampling_preds)
                        sampling_scores = []
                        sampling_preds = []
                        sample_idx = sample["sample_idx"][j]
                    answer = sample['answer'][j][k].split("</think>")[-1]
                    
                    result = extract_pred_and_parse(answer, data_name)
                    performance = per_sample_verification(result, sample['gt'])
                    result = [str(r) for r in result] 
                    # reasoning_preds.append(result)
                    # reasoning_scores.append(performance)
                    n_tokens.append(len(tokenizer(sample['answer'][j][k]).input_ids))
                    accs.append(1 if True in performance else 0)
                    sampling_scores.append(result)
                    sampling_preds.append(performance)
                
            scores.append(sampling_scores)
            preds.append(sampling_preds)
            

            sample.update({
                "score": scores,
                "final_answer": preds
            })

            updated_samples.append(sample)
        
        # analysis(samples)
        print(np.mean(accs))
        print(np.mean(n_tokens))
        try:
            save_jsonl(samples, generated_dataset_file)
        except Exception as e:
            print(f"Error saving generated reasoning: {e}")


if __name__ == "__main__":
    args = parse_args()
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    set_seed(args.seed)
    setup(args)