import argparse
import torch
import torch.nn.functional as F
from transformers import (
    AutoConfig, AutoTokenizer, AutoModel,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor
)
import json
import numpy as np
from tqdm import tqdm
import re
import os
import random
from qwen_vl_utils import process_vision_info

# ------------------------
# Reproducibility setup
# ------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # deterministic cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# -------------------------------------------------------
# SigLIP Similarity Model (for Open-World)
# -------------------------------------------------------
class SigLIPTextSimilarity:
    def __init__(self, model_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = AutoConfig.from_pretrained(model_name)
        config.max_position_embeddings = 64
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name, config=config, ignore_mismatched_sizes=True
        ).to(self.device)

    def get_text_embeddings(self, text):
        if isinstance(text, str):
            text = [text]
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            features = self.model.get_text_features(**inputs)

        return features / features.norm(dim=-1, keepdim=True)

    def forward(self, text1, text2):
        emb1 = self.get_text_embeddings(text1)
        emb2 = self.get_text_embeddings(text2)
        return F.cosine_similarity(emb1, emb2).item()


# -------------------------------------------------------
# Helper: extract <answer> ... </answer>
# -------------------------------------------------------
def extract_answer(output_str):
    match = re.search(r'<answer>(.*)</answer>', output_str, re.DOTALL)
    return match.group(1).strip() if match else None


# -------------------------------------------------------
# Closed-world Multiple-choice Evaluation
# -------------------------------------------------------
def eval_closedworld(model, processor, device, args):
    with open(args.prompt_path, "r") as f:
        data = [json.loads(line) for line in f]

    QUESTION_TEMPLATE = (
        "Given the question: {Question}, based on the options provided in {Options}, "
        "output the thinking process in <think> </think> and final choice in <answer> </answer> tags. "
        "The response format should be: <think>...</think> <answer>choice</answer>."
    )

    messages = []
    for item in data:
        messages.append([{
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{item['image_path']}"},
                {"type": "text", "text": QUESTION_TEMPLATE.format(
                    Question=item['question'], Options=item['options']
                )}
            ]
        }])

    all_outputs = []

    for i in tqdm(range(0, len(messages), args.batch_size)):
        batch = messages[i:i+args.batch_size]

        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in batch]

        image_inputs, video_inputs = process_vision_info(batch)

        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(device)

        # --- 单GPU生成，删除 generator 参数 ---
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

        trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, out)
        ]

        texts = processor.batch_decode(trimmed, skip_special_tokens=True)
        all_outputs.extend(texts)

    # ---- compute accuracy ----
    correct = 0
    results = []

    for item, output in zip(data, all_outputs):
        model_ans = extract_answer(output)
        gt = str(item["ground_truth"]).lower().replace("_", " ")

        if model_ans:
            model_ans = model_ans.lower().replace("_"," ")
            if gt in model_ans:
                correct += 1

        results.append({
            "question": item,
            "ground_truth": item["ground_truth"],
            "model_output": output,
            "parsed_answer": model_ans
        })

    acc = correct / len(data) * 100
    print(f"[Closed-world] Accuracy = {acc:.2f}%")

    # save
    with open(args.output_path, "w") as f:
        json.dump({
            "mode": "closedworld",
            "accuracy": acc,
            "results": results,
        }, f, indent=2)


# -------------------------------------------------------
# Open-world fine-grained category evaluation
# -------------------------------------------------------
def eval_openworld(model, processor, device, args):
    with open(args.prompt_path, "r") as f:
        data = [json.loads(line) for line in f]

    TEMPLATE = (
        "Given the question: {Question}. This is a fine-grained question. "
        "Output reasoning in <think></think> and the final answer in <answer></answer>."
    )

    siglip = SigLIPTextSimilarity(args.siglip_path)

    messages = []
    for item in data:
        messages.append([{
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{item['image_path']}"},
                {"type": "text", "text": TEMPLATE.format(Question=item['question'])}
            ]
        }])

    all_outputs = []

    for i in tqdm(range(0, len(messages), args.batch_size)):
        batch = messages[i:i+args.batch_size]
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in batch]

        image_inputs, video_inputs = process_vision_info(batch)
        inputs = processor(
            text=text, images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(device)

        # --- 单GPU生成，删除 generator 参数 ---
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

        trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, out)
        ]
        texts = processor.batch_decode(trimmed, skip_special_tokens=True)
        all_outputs.extend(texts)

    results = []
    correct = 0
    ss_scores = []

    for item, output in zip(data, all_outputs):
        gt = item['ground_truth']
        model_ans = extract_answer(output)
        base_output = model_ans if model_ans else output

        # detect domain by path
        path = item["image_path"]
        if "aircraft" in path:
            standard = "aircraft"
        elif "birds" in path:
            standard = "bird"
        elif "cars" in path:
            standard = "car"
        elif "flower" in path:
            standard = "flower"
        else:
            standard = "object"

        # semantic similarity
        ss = siglip.forward(base_output, gt)
        std_ss = siglip.forward(standard, gt)
        norm_ss = max(0.0, (ss - std_ss) / (1 - std_ss))
        ss_scores.append(norm_ss)

        # Text Include match
        if standard == "car":
            if " ".join(gt.lower().split()[:-2]) in base_output.lower():
                correct += 1
        else:
            if gt.lower().replace("_"," ") in base_output.lower():
                correct += 1

        results.append({
            "ground_truth": gt,
            "model_output": output,
            "answer": model_ans,
            "semantic_score": norm_ss,
            "standard": standard
        })

    acc = correct / len(data) * 100
    avg_ss = np.mean(ss_scores)

    print(f"[Open-world] TI accuracy = {acc:.2f}%")
    print(f"[Open-world] Avg Semantic Similarity = {avg_ss:.4f}")

    with open(args.output_path, "w") as f:
        json.dump({
            "mode": "openworld",
            "accuracy_TI": acc,
            "avg_semantic_similarity": avg_ss,
            "results": results
        }, f, indent=2)

# -------------------------------------------------------
# Main
# -------------------------------------------------------
def load_model(model_path, device):
    # 建议严格复现时，不使用 flash_attention_2，改用默认实现
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map=device
    )

    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = 'left'
    return model, processor


def build_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["open", "closed"], required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--prompt_path", required=True)
    parser.add_argument("--output_path", required=True)

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=2048)

    # required only in openworld mode
    parser.add_argument("--siglip_path", default="google/siglip-base-patch16-256")

    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    # Ensure output directory exists
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    # Build final output filename using model name, dataset name, and mode
    model_name = os.path.basename(args.model_path.strip("/"))
    dataset_name = os.path.basename(args.prompt_path.strip("/")).replace(".jsonl", "")
    mode_name = args.mode

    # Construct final output path
    final_output_file = f"{model_name}_{dataset_name}_{mode_name}.json"
    final_output_path = os.path.join(args.output_path, final_output_file)

    print(f"Results will be saved to: {final_output_path}")

    # Load model and processor
    model, processor = load_model(args.model_path, args.device)

    # Replace args.output_path so evaluation functions save to final path
    args.output_path = final_output_path

    # Run evaluation
    if args.mode == "closed":
        eval_closedworld(model, processor, args.device, args)

    elif args.mode == "open":
        eval_openworld(model, processor, args.device, args)

    else:
        raise ValueError("Invalid mode. Choose 'open' or 'closed'.")
