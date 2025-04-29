from collections import defaultdict
import argparse
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import random
import re

import numpy as np
import torch

from tqdm import tqdm
import json
from utils.data import load_generated_data, save_jsonl, load_jsonl
import re

# def remove_boxed(text):
#     # This matches \boxed{...} only if the content is non-empty
#     return re.sub(r'\\boxed{([^}]+)}', r'\1', text)

def set_seed(seed: int = 42) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    print(f"Random seed set as {seed}")

def random_wrong_answer(c_answer):
    prob = random.random()
    if prob < .25:
        return c_answer + 1
    elif prob < .5:
        return c_answer - 1
    elif prob < .75:
        return c_answer + 10
    else:
        return c_answer - 10

def select_samples(samples, args):
   
    selected_samples = []
    count = 0
    for sample in tqdm(samples):
        idx = 0
        question = sample['prompt']
        sample_idx = sample['sample_idx']
        split_reasoning = []
        final_answers = []
        for f in sample["final_answer"]:
            final_answers.extend(f)
        
        # first_reasoning_idx = 0
       
        for i, reasoning in enumerate(sample['reasonings']):
            if i + 1 == len(sample_idx) or sample_idx[i + 1] != idx:
                acc = final_answers[i]
                answer = sample['answer'][i][0].split('</think>')[-1]
                # first_reasoning = sample['reasonings'][first_reasoning_idx]
                
                if True in acc:
                    count += 1
                    new_first_reasoning = answer.replace(sample['gt'], str(random_wrong_answer(int(sample['gt']))))
                    # new_first_reasoning = answer
                    new_sample = {
                        "question": sample["question"],
                        "gt": sample['gt'],
                        "prompt": sample['prompt'],
                        "reasonings": [new_first_reasoning]
                    }
                    selected_samples.append(new_sample)
                    idx += 1
                    # first_reasoning_idx = i + 1
                    break
        

    # Save
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    generated_dataset_file = os.path.join(args.output_dir, "aime24_wrong_first_reasoning.json")
    print("sample number", len(selected_samples))
    print(f"Save to {generated_dataset_file}")
    json.dump(selected_samples, open(generated_dataset_file, "w",), indent=2)
    #save_jsonl(updated_samples, generated_dataset_file)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="/projects/0/gusr0608/self_correction_llms/outputs/aime24_outputs/test_DeepSeek-R1-Distill-Qwen-7B_seed0_num-1s0e-1_dataset_judge_answer.jsonl", type=str)
    parser.add_argument("--model_name_or_path", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", type=str)
    parser.add_argument("--output_dir", default="/projects/0/gusr0608/self_correction_llms/modified_first_reasoning", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    set_seed(args.seed)
    examples = list(load_jsonl(args.dataset_dir))
    # analysis(examples, args)
    select_samples(examples, args)