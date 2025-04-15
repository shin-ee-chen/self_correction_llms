from collections import defaultdict
import argparse
import os
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util

from utils.data import load_generated_data, save_jsonl, load_jsonl

def split_text_per_reason(text):
    # pattern = r'\b(wait|alternatively|however)\b'
    pattern = r'\b(alternatively)\b'
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    return [parts[i-1].strip() + parts[i].strip() if i-1 >= 0 else parts[i].strip() 
            for i in range(0, len(parts), 2) if parts[i].strip()]

def split_text_per_token(text):
    first_reason = split_text_per_reason(text)[0]
    chunk_len = len(first_reason.split(" "))
    text_tokens = text.split(" ")
    num_chunk = len(text_tokens) // chunk_len
    reasoning_chunks = []
    for i in range(num_chunk):
        chunk = text_tokens[i * chunk_len : (i + 1) * chunk_len]
        reasoning_chunks.append(" ".join(chunk))
    
    return reasoning_chunks



def plot_similarity_vs_position(similarities, save_path=None):
    """
    Plot similarity scores versus reasoning position (index-based).

    Args:
        similarities (list of float): Similarity scores where index = reasoning position.
        save_path (str, optional): Path to save the plot (e.g., 'plot.pdf'). If None, displays the plot.
    """
    positions = list(range(len(similarities)))

    plt.figure(figsize=(7, 4))
    plt.plot(positions, similarities, marker='o', linestyle='-', color='#0072B2', linewidth=2)
    
    plt.xlabel('Reasoning Position in Think Context', fontsize=12)
    plt.ylabel('Similarity Score', fontsize=12)
    
    if len(positions) < 15:
        plt.xticks(positions)
    else:
        plt.xlim(0, len(positions))
        plt.xticks(range(0, len(positions) + 1, 5))  # show ticks every 5 positions
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

    

def set_seed(seed: int = 42) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    print(f"Random seed set as {seed}")

def get_reasoning_answer_similarity(reasonings, answers, embed_model, output_dir, split_type="reason"):
    sep_reasonings = []
    answers_sep = []
    sep_idx = []
    answer_embedding = embed_model.encode(answers)

    for i in range(len(reasonings)):
        reasoning = split_text_per_reason(reasonings[i]) if split_type == "reason" else split_text_per_token(reasonings[i])
        for j, r in enumerate(reasoning):
            sep_reasonings.append(r)
            answers_sep.append(answer_embedding[i])
            sep_idx.append(i)
    
    reasonings_embedding = embed_model.encode(sep_reasonings)
   
    
    similarity_scores = defaultdict(list)
    for i, r_emb in enumerate(reasonings_embedding):
        similarity = util.cos_sim(answers_sep[i],r_emb)
        similarity_scores[sep_idx[i]].append(similarity)
    
    highest_score = -1
    idx = -1

    # longest_len = max(len(lst) for lst in list(similarity_scores.values()))
    lengths = list(map(len, list(similarity_scores.values())))
    longest_length = max(lengths)
    mean_length = int(np.round(np.mean(lengths), 0).item())
    median_length = int(np.median(lengths).item())
    
    inte_lens = {"longest_length":longest_length, 
                 "mean_length": mean_length, 
                 "median_length": median_length}
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for len_type, inte_len in inte_lens.items():
        similarities = []
        # print("inte_len:", inte_len)
        pos_sim = defaultdict(list)
        for _, v in similarity_scores.items():
            reasonings_score = [s.item() for s in v]
            inte_v = interpolate_list(reasonings_score, inte_len)
            for i, sim in enumerate(inte_v):
                pos_sim[i].append(sim)

        for k, v in pos_sim.items():
            # print(np.round(np.mean(v), 2), end = " ")
            similarities.append(np.round(np.mean(v), 2))
            # if highest_score < np.round(np.mean(v), 2):
            #     highest_score = np.round(np.mean(v), 2)
            #     idx = k
    
        # print("highest", idx, highest_score)
        plot_similarity_vs_position(similarities, 
                                    save_path=os.path.join(output_dir, f"{split_type}_{len_type}.pdf"))
    


def interpolate_list(x, target_len):
    x = np.array(x)
    original_indices = np.linspace(0, 1, len(x))
    target_indices = np.linspace(0, 1, target_len)
    interpolated = np.interp(target_indices, original_indices, x)
    return interpolated.tolist()


def analysis(samples, args):
    correct_r = defaultdict(list)
    wrong_r = defaultdict(list)
    tokenizer =  AutoTokenizer.from_pretrained(args.model_name_or_path)
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    reasoning_similarity = defaultdict(list)
    
    reasonings = []
    answer_think_sum = []
    for sample in samples:
        sample_idx = 0 
        first_correct = {"f_acc": [], "r_acc":[], "n_tokens":[], "n_reasons":[], "count": 0}
        first_wrong = {"f_acc": [], "r_acc":[], "n_tokens":[], "n_reasons":[], "count": 0}
        
        for i in range(len(sample["final_answer"])):
            is_first_correct = True if True in sample["final_answer"][i][0] else False
            type = first_correct if is_first_correct else first_wrong
            type["count"] += 1
            type["n_reasons"].append(len(sample["final_answer"][i]))
            type["f_acc"].append(1 if True in sample["final_answer"][i][-1] else 0)
            r_acc = []
            for j, ans in enumerate(sample["final_answer"][i][1:]):
                r_acc.append(1 if True in ans else 0)
            type["r_acc"].append(np.mean(r_acc))
            
            begin_idx = -1
            for j, reasoning in enumerate(sample["reasonings"]):
                if sample["sample_idx"][j] == sample_idx:
                   begin_idx = j if begin_idx < 0 else begin_idx
                   if j+1 >= len(sample["sample_idx"]) or \
                        sample["sample_idx"][j+1] != sample_idx:
                            tokens = tokenizer(reasoning).input_ids
                            type["n_tokens"].append(len(tokens))
                            reasonings.append(reasoning)
                            answer_think_sum.append(sample["answer"][j][0])
                            sample_idx += 1
                            break
            
        for k, correct_v in first_correct.items():
            wrong_v = first_wrong[k]
            if isinstance(correct_v, list):
                correct_r[k].append(np.nanmean(correct_v))
                wrong_r[k].append(np.nanmean(wrong_v))
            else:
                correct_r[k].append(correct_v)
                wrong_r[k].append(wrong_v)
            
        
    for k, correct_v in correct_r.items():
        print("Metric", "First Correct", "First Wrong")
        wrong_v = wrong_r[k]
        print(k, np.round(np.nanmean(correct_v), 2), np.round(np.nanmean(wrong_v), 2))
    
    for k, v in reasoning_similarity.items():
        print(k, np.round(np.mean(v), 2), end="\t")
    
    print("Split with reason")
    get_reasoning_answer_similarity(reasonings, answer_think_sum, embed_model, args.output_dir, split_type="reason")
    print("Split with token")
    get_reasoning_answer_similarity(reasonings, answer_think_sum, embed_model, args.output_dir, split_type="token")
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="/projects/0/gusr0608/outputs/output/math/judge/test_DeepSeek-R1-Distill-Qwen-7B_seed0_num-1s0e-1_dataset_judge_answer.jsonl", type=str)
    parser.add_argument("--model_name_or_path", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", type=str)
    parser.add_argument("--output_dir", default="/projects/0/gusr0608/self_correction_llms/similarity_plots", type=str)
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
    analysis(examples, args)