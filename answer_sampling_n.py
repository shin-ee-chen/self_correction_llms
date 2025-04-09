import os
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
import re

import torch
from vllm import LLM, SamplingParams

from utils.data import load_generated_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="math500", type=str)
    parser.add_argument("--dataset_dir", default="/projects/0/gusr0608/self_correction_llms/outputs/outout/math500/first_reasonings/test_DeepSeek-R1-Distill-Qwen-7B_seed0_t1_len9800_num2s0e-1_first_reasoning_dataset.json", type=str)
    parser.add_argument("--model_name_or_path", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", type=str)
    parser.add_argument("--output_dir", default="/projects/0/gusr0608/self_correction_llms/outputs", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--min_p", default=0., type=float)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--max_num_seqs", type=int, default=32)
    parser.add_argument('--max_model_len', type=int, default=64000)
    parser.add_argument("--n_sampling", default=1, type=int, help="I.e. n")
    args = parser.parse_args()
    # top_p must be 1 when using greedy sampling (vllm)
    args.top_p = 1 if args.temperature == 0 else args.top_p
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
    # sample `num_test_sample` from dataset for debug purpose
    if args.num_test_sample > 0:
        examples = examples[: args.num_test_sample]
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # Get output file name
    model_name = args.model_name_or_path.split('/')[-1]
    out_file_prefix = f"{args.split}_{model_name}_seed{args.seed}_t{args.temperature}_len{args.max_tokens_per_call}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    generated_dataset_file = f"{output_dir}/{data_name}/predictions/{out_file_prefix}_num{args.num_test_sample}s{args.start}e{args.end}_dataset_predictions.json"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)
    os.makedirs(f"{output_dir}/{data_name}/predictions", exist_ok=True)
    return examples, generated_dataset_file


def setup(args):
    # Load model
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        trust_remote_code=True,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        seed=args.seed,
    )
    tokenizer = llm.get_tokenizer()

    # Infer & eval
    data_paths = args.dataset_dir.split(",")
    data_list = args.data_names.split(",")
    assert len(data_list) == len(data_paths)

    for i, data_name in enumerate(data_list):
        main(llm, tokenizer, data_name, data_paths[i], args)


def split_text(text):
    # pattern = r'\b(wait|alternatively|however)\b'
    pattern = r'\b(alternatively)\b'
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    return [parts[i-1].strip() + parts[i].strip() if i-1 >= 0 else parts[i].strip() 
            for i in range(0, len(parts), 2) if parts[i].strip()]
    

def process_reasonings(reasonings):
    split_reasons = split_text(reasonings)

    reason = split_reasons[0]

    seperate_reasonings = [reason]
    for i in range(1, len(split_reasons)):
        reason += " " + split_reasons[i]
        seperate_reasonings.append(reason)
    return seperate_reasonings

def main(llm, tokenizer, data_name, data_path, args):
    examples, generated_dataset_file = prepare_data(data_path, data_name, args)
    print("=" * 50)
    print("data:", data_name, " , #samples:", len(examples))

    samples = []
    for i, example in tqdm(enumerate(examples), total=len(examples)):
        reasonings = []
        sample_idx = []
        for idx, r in enumerate(example['reasonings']):
            split_reasonings = process_reasonings(r)
            reasonings.extend(split_reasonings)
            sample_idx.extend([idx] * len(split_reasonings))

        sample = {
            "question": example["question"],
            "gt": example['gt'],
            "prompt": example['prompt'],
            "reasonings": reasonings,
            "sample_idx": sample_idx
            }
        samples.append(sample)

    prompts = []
    for i, sample in enumerate(samples):
        for j in range(len(sample['reasonings'])):
            for _ in range(args.n_sampling):
                prompts.append(sample['prompt'] + sample['reasonings'][j] + "\n</think>\n\n")

    # Start inference
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        min_p=args.min_p,
        max_tokens=args.max_tokens_per_call,
        min_tokens=2,
        n=1,
        skip_special_tokens=False,
        seed=args.seed,

    )
    outputs = llm.generate(prompts, sampling_params)
    outputs = sorted(outputs, key=lambda x: int(x.request_id))
    generated_reasonings = [output.outputs[0].text for output in outputs]
    assert len(generated_reasonings) == len(prompts)

    answers = []
    start_idx = 0
    for i, sample in enumerate(samples):
        # n_first_reasoning_sampling = len(sample['first_reasonings'])
        n_reasoning_sampling = len(sample['reasonings'])
        sample_answers = generated_reasonings[start_idx : start_idx + (n_reasoning_sampling * args.n_sampling)]
        assert len(sample_answers) == n_reasoning_sampling * args.n_sampling
        answers.append([sample_answers[i * args.n_sampling:(i + 1) * args.n_sampling] for i in range(n_reasoning_sampling)])
        start_idx = start_idx + (n_reasoning_sampling * args.n_sampling)

    updated_samples = []
    for i, sample in enumerate(samples):
        sample.update({"answer": answers[i]})
        updated_samples.append(sample)

    #save_jsonl(updated_samples, generated_dataset_file)
    print(f"Save to {generated_dataset_file}")
    json.dump(updated_samples, open(generated_dataset_file, "w",), indent=2)


if __name__ == "__main__":
    args = parse_args()
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    set_seed(args.seed)
    setup(args)