import os
import gc
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict


# ======================
# 路径配置
# ======================
REVIEW_PATH = "/root/autodl-tmp/DEP-main/datasets/coldstart_reviews.json"
OUTPUT_DIR = "/root/autodl-tmp/DEP-main/datasets/embeddings_yelp"

MODEL_NAME = "BAAI/bge-m3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CACHE_DIR = "/root/autodl-tmp/DEP-main/model_point"

# ======================
# 加载模型与分词器（指定缓存目录）
# ======================
print(f"Loading embedding model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
model = AutoModel.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, device_map="auto")
model.eval()


# ======================
# 工具函数
# ======================
@torch.no_grad()
def get_embeddings(texts, batch_size=64):
    """将文本批量编码为embedding"""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch_txts = texts[i:i + batch_size]
        batch = tokenizer(batch_txts, truncation=True, padding=True, return_tensors="pt")
        for k in batch:
            batch[k] = batch[k].to(model.device)
        outputs = model(**batch)
        embs = torch.nn.functional.normalize(outputs.last_hidden_state[:, 0], p=2, dim=1)
        all_embs.append(embs.detach().cpu())
        del batch, outputs, embs
        torch.cuda.empty_cache()
        gc.collect()
    return torch.cat(all_embs, dim=0)


# ======================
# Step 1. 加载 Yelp 评论数据
# ======================
print(f"Loading Yelp coldstart dataset from {REVIEW_PATH}...")
with open(REVIEW_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total records loaded: {len(data)}")

# 分组结构： {center_user_id: {user_id: [ {text, date}, ... ] }}
grouped_reviews = defaultdict(lambda: defaultdict(list))

for r in tqdm(data, desc="Grouping reviews"):
    cid = r["center_user_id"]
    uid = r["user_id"]
    text = r["text"].strip()
    date = r.get("date", "")
    if text:
        grouped_reviews[cid][uid].append({"text": text, "date": date})

# ✅ Step 1.1 对每个用户的评论按时间升序排序
for cid in grouped_reviews:
    for uid in grouped_reviews[cid]:
        grouped_reviews[cid][uid] = sorted(
            grouped_reviews[cid][uid],
            key=lambda x: x["date"]
        )


# ======================
# Step 2. 遍历每个中心用户，分别为ego网络中的每个用户生成embedding
# ======================
for cid, user_reviews in tqdm(grouped_reviews.items(), desc="Processing center users"):
    center_dir = os.path.join(OUTPUT_DIR, cid)
    os.makedirs(center_dir, exist_ok=True)

    for uid, reviews in user_reviews.items():
        if len(reviews) == 0:
            continue

        save_path = os.path.join(center_dir, f"{uid}.emb")
        if os.path.exists(save_path):
            continue

        try:
            # ✅ 按时间顺序提取文本
            texts = [r["text"] for r in reviews]
            embs = get_embeddings(texts, batch_size=64)
            torch.save(embs, save_path)
        except Exception as e:
            print(f"[warning] failed to embed {uid}: {e}")
            continue

print("\n✅ All embeddings have been generated successfully (with time-sorted reviews)!")
print(f"Saved under directory → {OUTPUT_DIR}")
