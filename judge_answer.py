import os
import time
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch
from utils.data import load_generated_data, save_jsonl
from utils.parser import deserialize_list_of_lists, extract_pred_and_parse
from utils.eval import per_sample_verification

from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="math", type=str)
    parser.add_argument("--dataset_dir", default="/projects/0/gusr0608/self_correction_llms/outputs/math500/predictions/test_DeepSeek-R1-Distill-Qwen-7B_seed0_t0_len2048_num-1s0e-1_dataset_predictions.json", type=str)
    parser.add_argument("--model_name_or_path", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
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

def analysis(samples):
    correct_r = defaultdict(list)
    wrong_r = defaultdict(list)
    
    for sample in samples:
        sample_idx = 0 
        first_correct = {"f_acc": [], "n_tokens":[], "n_reasons":[]}
        first_wrong = {"f_acc": [], "n_tokens":[], "n_reasons":[]}
        
        for i in range(len(sample["final_answer"])):
            is_first_correct = True if True in sample["final_answer"][i][0] else False
            type = first_correct if is_first_correct else first_wrong
            
            type["n_reasons"].append(len(sample["final_answer"][i]))
            type["f_acc"].append(1 if True in sample["final_answer"][i][-1] else 0)
            
               
            for j, reasoning in enumerate(sample["reasonings"]):
                if sample["sample_idx"][j] == sample_idx:
                   if j+1 >= len(sample["sample_idx"]) or \
                        sample["sample_idx"][j+1] != sample_idx:
                            type["n_tokens"].append(len(reasoning.split()))
                            sample_idx += 1
                            break
            
        for k, correct_v in first_correct.items():
            wrong_v = first_wrong[k]
            correct_r[k].append(np.nanmean(correct_v))
            wrong_r[k].append(np.nanmean(wrong_v))
        
    for k, correct_v in correct_r.items():
        print("Metric", "First Correct", "First Wrong")
        wrong_v = wrong_r[k]
        print(k, np.round(np.nanmean(correct_v), 2), np.round(np.nanmean(wrong_v), 2))
                
        
def main(data_name, data_path, args):
        examples, generated_dataset_file = prepare_data(data_path, data_name, args)
        print("Data prepration done!")
        print("=" * 50)
        print("data:", data_name, " , #samples:", len(examples))

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


        n_reasoning_sampling = len(samples[0]['reasonings'])
        n_answers_sampling = len(samples[0]['answer'][0])

        updated_samples = []
        
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
                    
                    result = extract_pred_and_parse(sample['answer'][j][k], data_name)
                    performance = per_sample_verification(result, sample['gt'])
                    result = [str(r) for r in result] 
                    # reasoning_preds.append(result)
                    # reasoning_scores.append(performance)
                    
                    sampling_scores.append(result)
                    sampling_preds.append(performance)
                
            scores.append(sampling_scores)
            preds.append(sampling_preds)

            sample.update({
                "score": scores,
                "final_answer": preds
            })

            updated_samples.append(sample)
        
        analysis(samples)
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