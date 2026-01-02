import os
import torch
import json
import evaluate
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from bert_score import score as bert_score
from models.model_eval import DEPModel
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from models.gcn_gib import SocialGCN_GBSR
from torch_geometric.data import Data
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["infer", "eval"], default="eval", required=True, help="运行模式：infer 或 eval")
parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp/my_project/data/processed_test",
                    help="测试数据路径")
parser.add_argument("--output_dir", type=str, default="output", help="输出文件夹路径")
parser.add_argument("--llm_backbone", type=str,
                    default="/root/autodl-tmp/my_project/model_point/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
                    help="主干模型（未训练的LLM）路径")
parser.add_argument("--ckpt_dir", type=str,
                    default="/root/autodl-tmp/my_project/output/checkpoint-1125",
                    help="训练后保存的checkpoint文件夹")
args = parser.parse_args()

args.ckpt_path = os.path.join(args.ckpt_dir, "trainable_modules.pt")
os.makedirs(args.output_dir, exist_ok=True)

def postprocess_output(text):
    text = text.strip()
    text = text.replace("\r", " ").replace("\n", " ")
    text = " ".join(text.split())     # 去掉连续空格
    return text

print("🔧 Loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(args.ckpt_dir, trust_remote_code=True)

print(f"🔧 Loading base model from: {args.llm_backbone}")
personal_model = DEPModel.from_pretrained(
    pretrained_model_name_or_path=args.llm_backbone,
    tokenizer=tokenizer,
    torch_dtype=torch.float16, 
    attn_implementation="flash_attention_2",
    training=False
).to("cuda")

if os.path.exists(args.ckpt_path):
    print(f"Loading trained parameters from {args.ckpt_path} ...")
    state_dict = torch.load(args.ckpt_path, map_location="cpu")

    model_dtype = next(personal_model.parameters()).dtype
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            state_dict[k] = v.to(dtype=model_dtype)

    missing, unexpected = personal_model.load_state_dict(state_dict, strict=False)
    print(f"已加载训练后参数: {len(state_dict)} tensors (cast to {model_dtype})")
    if missing:
        print(f"Missing keys: {missing[:5]} ...")
    if unexpected:
        print(f"Unexpected keys: {unexpected[:5]} ...")
else:
    print(f"未找到训练后参数文件: {args.ckpt_path}")

personal_model.eval()
print("模型加载完成，可开始推理或评估！")

print(f"Loading dataset from {args.data_dir}")
dataset = load_from_disk(args.data_dir)
print(f"Loaded {len(dataset)} samples")

def custom_collate_fn(batch):
    collated = {}
    collated["graph_data"] = [b.get("graph_data", None) for b in batch]
    if "embeddings" in batch[0]:
        emb_list = []
        for b in batch:
            emb = b["embeddings"]
            if isinstance(emb, list):
                emb = torch.tensor(emb, dtype=torch.float32)
            elif isinstance(emb, torch.Tensor):
                emb = emb.to(dtype=torch.float32)
            else:
                raise TypeError(f"embeddings type not supported: {type(emb)}")
            emb_list.append(emb)
        collated["embeddings"] = torch.stack(emb_list, dim=0)

    keys = set().union(*[b.keys() for b in batch])
    keys.discard("graph_data")
    keys.discard("embeddings")
    for k in keys:
        v0 = batch[0][k]
        if isinstance(v0, torch.Tensor):
            collated[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            collated[k] = [b[k] for b in batch]
    return collated

if args.mode == "infer":
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)
    predictions = []

    print("Start inference ...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            graph_data = batch.get("graph_data", None)
            embeddings = batch.get("embeddings", None)
            inp_str = batch.get("inp_str", None)

            texts = personal_model.generate_text(
                graph_data=graph_data,
                embeddings=embeddings,
                inp_str=inp_str,
                max_new_tokens=2048,
                temperature=0.7,
                top_p=0.9,
                training=False,
            )

            predictions.extend(texts)

    # === 保存预测结果 ===
    output_path = os.path.join(args.output_dir, "predictions-justhavefriendsreviewandweight.txt")
    print(f"Saving predictions to {output_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        for pred in predictions:

            f.write(pred.strip() + "\n---------------------------------\n")

    print("Saved all predictions!")

elif args.mode == "eval":
    #predictions_path = os.path.join(args.output_dir, "predictions.txt")
    predictions_path = "/root/autodl-tmp/my_project/output/predictions-justhavefriendsreviewandweight.txt"

    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"{predictions_path} not found. 请先运行 `--mode infer` 生成预测结果。")

    print(f"Loading predictions from {predictions_path}")
    with open(predictions_path, "r", encoding="utf-8") as f:
        blocks = f.read().strip().split("---------------------------------")
        predictions = [b.strip() for b in blocks if b.strip()]


    references = list(dataset["out_str"])


    print("Evaluating results ...")
    bleu_metric = evaluate.load("sacrebleu")
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")

    result_bleu = bleu_metric.compute(predictions=predictions, references=[[ref] for ref in references])
    result_rouge = rouge_metric.compute(predictions=predictions, references=references)
    result_meteor = meteor_metric.compute(predictions=predictions, references=references)

    P, R, F1 = bert_score(
        predictions, references, model_type="allenai/led-base-16384", lang="en", verbose=False
    )

    result = {
        "rouge-1": float(result_rouge["rouge1"]),
        "rouge-L": float(result_rouge["rougeL"]),
        "meteor": float(result_meteor["meteor"]),
        "bleu": float(result_bleu["score"]),
        "bertscore": float(F1.mean()),
    }

    print("\nEvaluation Results:")
    for k, v in result.items():
        print(f"{k:<10}: {v:.4f}")

    # === 保存结果到 JSON ===
    result_path = os.path.join(args.output_dir, "eval_results-justhavefriendsreviewandweight.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
    print(f"Results saved to {result_path}")
