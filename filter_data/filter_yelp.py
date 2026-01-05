import json
import os
import time
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# ======================
# 路径配置
# ======================
BASE_DIR = r"your_path"
OUT_DIR = r"your_path"

USER_PATH = os.path.join(BASE_DIR, "yelp_academic_dataset_user.json")
REVIEW_PATH = os.path.join(BASE_DIR, "yelp_academic_dataset_review.json")
BUSINESS_PATH = os.path.join(BASE_DIR, "yelp_academic_dataset_business.json")

OUT_TEXT_JSON = os.path.join(OUT_DIR, "coldstart_reviews.json") 
OUT_GRAPH_NPY = os.path.join(OUT_DIR, "social_graphs.npy")

MAX_USERS = 4000
def load_json_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def parse_friends_field(raw):
    if not raw or raw == "None":
        return set()
    if isinstance(raw, list):
        return set(raw)
    return set(x for x in raw.split(", ") if x)


def parse_categories_field(raw):
    if not raw or raw == "None":
        return []
    if isinstance(raw, list):
        return raw
    return [x for x in raw.split(", ") if x]

print("Loading users...")
t0 = time.time()
user_friends = {}
user_review_count = {}

for u in tqdm(load_json_lines(USER_PATH), desc="users"):
    uid = u["user_id"]
    friends_set = parse_friends_field(u.get("friends"))
    user_friends[uid] = friends_set
    user_review_count[uid] = int(u.get("review_count", 0))

print(f"Total users loaded: {len(user_friends)}  |  {time.time()-t0:.1f}s")

print("Loading businesses...")
t1 = time.time()

business_info = {}   # 保存 name + categories + attributes

for b in tqdm(load_json_lines(BUSINESS_PATH), desc="businesses"):
    bid = b["business_id"]
    business_info[bid] = {
        "name": b.get("name", ""),
        "categories": parse_categories_field(b.get("categories")),
        "attributes": b.get("attributes", {}) 
    }

print(f"Total businesses loaded: {len(business_info)}  |  {time.time()-t1:.1f}s")
print("Loading reviews...")
t2 = time.time()
user_reviews = defaultdict(list)

for r in tqdm(load_json_lines(REVIEW_PATH), desc="reviews"):
    uid = r["user_id"]
    text = r["text"].strip()
    if len(text) < 5:
        continue

    user_reviews[uid].append({
        "review_id": r["review_id"],
        "business_id": r["business_id"],
        "stars": r["stars"],
        "date": r["date"],
        "text": text
    })

print(f"Total users with valid reviews: {len(user_reviews)}  |  {time.time()-t2:.1f}s")

print("Selecting cold-start users (mutual friends only, each friend≥8 valid reviews)...")
t3 = time.time()

center_users = []
center_to_mutual_friends = {}

candidate_users = [uid for uid, rs in user_reviews.items() if 4 <= len(rs) <= 30]
print(f"Initial candidates: {len(candidate_users)}")

for uid in tqdm(candidate_users, desc="filtering centers"):
    friends = user_friends.get(uid, set())
    if not friends:
        continue

    if len(friends) > 200:
        continue
    mutual = {f for f in friends if uid in user_friends.get(f, ())}

    valid_mutual = set()
    total_reviews = 0

    for f in mutual:
        n_reviews = len(user_reviews.get(f, ()))
        if n_reviews >= 8:
            valid_mutual.add(f)
            total_reviews += n_reviews

    num_mutual = len(valid_mutual)
    if num_mutual < 3 or num_mutual > 6:
        continue

    if 64 <= total_reviews <= 250:
        center_users.append(uid)
        center_to_mutual_friends[uid] = valid_mutual

    if len(center_users) >= MAX_USERS:
        break

print(f"Eligible cold-start users found: {len(center_users)}  |  {time.time()-t3:.1f}s")

print("Building graphs and streaming ordered reviews...")
t4 = time.time()
os.makedirs(OUT_DIR, exist_ok=True)

graph_dict = {}

with open(OUT_TEXT_JSON, "w", encoding="utf-8") as f_out:
    for center_uid in tqdm(center_users, desc="center_graphs"):
        mutual_friends = center_to_mutual_friends[center_uid]

        ordered_nodes = [center_uid] + [f for f in mutual_friends if f != center_uid]
        edges = []
        for f in mutual_friends:
            edges.append((center_uid, f))
            edges.append((f, center_uid))

        for f1 in mutual_friends:
            inter = mutual_friends & user_friends.get(f1, set())
            for f2 in inter:
                if f1 != f2:
                    edges.append((f1, f2))

        ordered_edges = list(dict.fromkeys(edges))

        graph_dict[center_uid] = {"nodes": ordered_nodes, "edges": ordered_edges}

        for uid in ordered_nodes:
            for r in user_reviews.get(uid, []):
                bid = r["business_id"]
                biz = business_info.get(bid, {})

                rec = {
                    "center_user_id": center_uid,
                    "user_id": uid,
                    "is_center": int(uid == center_uid),
                    "business_id": bid,
                    "business_name": biz.get("name", ""),
                    "categories": biz.get("categories", []),
                    "stars": r["stars"],
                    "date": r["date"],
                    "text": r["text"]
                }
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"Reviews NDJSON written to: {OUT_TEXT_JSON}  |  {time.time()-t4:.1f}s")

t5 = time.time()
np.save(OUT_GRAPH_NPY, graph_dict, allow_pickle=True)
print(f"Saved social graphs → {OUT_GRAPH_NPY}  |  {time.time()-t5:.1f}s")

print("Finished: All reviews contain business_name.")
