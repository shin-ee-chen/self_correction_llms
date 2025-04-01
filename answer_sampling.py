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
from utils.data import save_jsonl, load_generated_data
from utils.parser import extract_pred_and_parse, serialize_list_of_lists
from utils.eval import per_sample_verification


def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--dataset_dir", default="", type=str)
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


def prepare_data(data_path, data_name, args):
    
    examples = load_generated_data(data_path)
    # sample `num_test_sample` from dataset for debug purpose
    if args.num_test_sample > 0:
        examples = examples[: args.num_test_sample]


    examples = examples[args.start : len(examples) if args.end == -1 else args.end]
    model_name = args.model_name_or_path.split('/')[-1]
    # get out_file name
    out_file_prefix = f"{args.split}_{model_name}_seed{args.seed}_t{args.temperature}_len{args.max_tokens_per_call}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    
    generated_dataset_file = f"{output_dir}/{data_name}/predictions/{out_file_prefix}_num{args.num_test_sample}s{args.start}e{args.end}_dataset_predictions.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)
    os.makedirs(f"{output_dir}/{data_name}/predictions", exist_ok=True)
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
    data_paths = args.dataset_dir.split(",")
    data_list = args.data_names.split(",")


    assert len(data_list) == len(data_paths)

    results = []
    for i, d in enumerate(data_list):
        main(llm, tokenizer, d, data_paths[i], args)


    
def main(llm, tokenizer, data_name, data_path, args):


        examples, generated_dataset_file = prepare_data(data_path, data_name, args)
        print("Data prepration done!")
        print("=" * 50)
        print("data:", data_name, " , #samples:", len(examples))
        if len(examples) > 0:
            print(examples[0])

        samples = []
        
        for i, example in tqdm(enumerate(examples), total=len(examples)):
            


            sample = {
                "question": example["question"],
                "gt": example['gt'],
                "prompt": example['prompt'],
                'first_reasonings': example['first_reasonings']
            }

            samples.append(sample)


        n_first_reasoning_sampling = len(samples[0]['first_reasonings'])

        prompts = []
        for i, sample in enumerate(samples):
            for j in range(n_first_reasoning_sampling):
                for _ in range(args.n_sampling):
                    prompts.append(sample['prompt']+ sample['first_reasonings'][j])
        # prompts = [sample["prompt"] for sample in samples for _ in range(args.n_sampling)]

        assert len(prompts) == len(samples) * n_first_reasoning_sampling * args.n_sampling

        for i, p in enumerate(prompts):
            if "</think>" not in p:
                prompts[i] = prompts[i] + "</think>"

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

        )

        outputs = llm.generate(prompts, sampling_params)
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        generated_reasonings = [output.outputs[0].text for output in outputs]

        assert len(generated_reasonings) == len(prompts)

        ### shape: n_first_reasoning_sampling * args.n_sampling
        answers = []
        for i, sample in enumerate(samples):
            tmp_answers = generated_reasonings[i * n_first_reasoning_sampling * args.n_sampling : (i + 1) * n_first_reasoning_sampling * args.n_sampling]
            assert len(tmp_answers) == n_first_reasoning_sampling * args.n_sampling
            answers.append([tmp_answers[i * args.n_sampling:(i + 1) * args.n_sampling] for i in range(n_first_reasoning_sampling)])

        updated_samples = []
        
        
        
        for i, sample in enumerate(samples):
            sample.update({
                "answer": answers[i],
             })
            updated_samples.append(sample)



        try:
            save_jsonl(updated_samples, generated_dataset_file)

        except Exception as e:
            print(f"Error saving generated reasoning: {e}")


if __name__ == "__main__":
    args = parse_args()
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    set_seed(args.seed)
    setup(args)