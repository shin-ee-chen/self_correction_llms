import os
import time
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch
from vllm import LLM, SamplingParams
import pandas as pd
from utils.data import load_data, construct_prompt, save_jsonl
from utils.parser import parse_question, parse_ground_truth, extract_pred_and_parse
from utils.eval import obtain_3d_sub_scores_and_preds, obtain_3d_scores, per_sample_verification


def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
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
    parser.add_argument('--enable_prefix_caching', action='store_true', default=False)
    parser.add_argument('--disable_chunked_prefill', action='store_true', default=False)
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


def prepare_data(data_name, args):
    if "math500_level" in data_name:
        level = int(data_name.strip()[-1])
        examples = load_data("math500", args.split, args.data_dir)
        examples = [example for example in examples if example["level"]==level]
    else:
        examples = load_data(data_name, args.split, args.data_dir)


    if args.num_test_sample > 0:
        examples = examples[: args.num_test_sample]

    # select start and end
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]
    model_name = args.model_name_or_path.split('/')[-1]
    # get out_file name
    out_file_prefix = f"{args.split}_{model_name}_seed{args.seed}_t{args.temperature}_len{args.max_tokens_per_call}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    
    generated_dataset_file = f"{output_dir}/{data_name}/first_reasonings/{out_file_prefix}_num{args.num_test_sample}s{args.start}e{args.end}_first_reasoning_dataset.json"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)
    os.makedirs(f"{output_dir}/{data_name}/first_reasonings", exist_ok=True)
    return examples, generated_dataset_file


def setup(args):

    
    # load model
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        trust_remote_code=True,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=args.enable_prefix_caching,
        enable_chunked_prefill=not args.disable_chunked_prefill,
        max_model_len=args.max_model_len,
        seed=args.seed,
    )
    tokenizer = llm.get_tokenizer()

    # infer & eval
    data_list = args.data_names.split(",")
    results = []
    for data_name in data_list:
        main(llm, tokenizer, data_name, args)


def main(llm, tokenizer, data_name, args):

    stop_token = ["Alternatively,"]

    examples, generated_dataset_file = prepare_data(data_name, args)
    print("Data prepration done!")
    print("=" * 50)
    print("data:", data_name, " , #samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    samples = []

    for i, example in tqdm(enumerate(examples), total=len(examples)):
        idx = example["idx"]

        # parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt = parse_ground_truth(example, data_name)
        full_prompt = construct_prompt(example, data_name, args)

        # if i == args.start:
            # print(full_prompt)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt": str(gt[0]),
            "prompt": full_prompt,
        }

        # add remain fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    # creating prompts
    prompts = [sample["prompt"] for sample in samples for _ in range(args.n_sampling)]
    questions = [sample["question"] for sample in samples for _ in range(args.n_sampling)]
    gt = [sample["gt"] for sample in samples for _ in range(args.n_sampling)]

    

    # start inference
    start_time = time.time()
    # Either load existing think answers or generate new ones
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        min_p=args.min_p,
        max_tokens=args.max_tokens_per_call,
        n=1,
        skip_special_tokens=False,
        seed=args.seed,
        stop=stop_token,
    )

    outputs = llm.generate(prompts, sampling_params)
    outputs = sorted(outputs, key=lambda x: int(x.request_id))
    generated_reasonings = [output.outputs[0].text for output in outputs]

    assert len(generated_reasonings) == len(prompts)

    
    data = {
        'question': questions,
        'gt': gt,
        'prompt':prompts,
        'first_reasoning': generated_reasonings,
        'dataset_name': [data_name] * len(questions)

    }


    try:
        
        with open(generated_dataset_file, 'w') as f:
            json.dump(data, f)
        print(f"Saved generated reasoings to {generated_dataset_file}")
    except Exception as e:
        print(f"Error saving generated reasoning: {e}")



if __name__ == "__main__":
    args = parse_args()
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    set_seed(args.seed)
    setup(args)