import os
import json
from collections import defaultdict
from tqdm import tqdm

BASE_DIR = r"your_path"
IN_PATH = os.path.join(BASE_DIR, "coldstart_reviews.json")
TRAIN_PATH = os.path.join(BASE_DIR, "train.json")
VAL_PATH = os.path.join(BASE_DIR, "val.json")
TEST_PATH = os.path.join(BASE_DIR, "test.json")


print(f"Loading dataset from {IN_PATH}...")
with open(IN_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
print(f"Total records: {len(data)}")

center_user_reviews = defaultdict(list)
for r in data:
    center_user_reviews[r["center_user_id"]].append(r)
print(f"Total center users: {len(center_user_reviews)}")

train_records, val_records, test_records = [], [], []

for cid, reviews in tqdm(center_user_reviews.items(), desc="Processing center users"):
    self_reviews = [r for r in reviews if r["user_id"] == cid]
    friend_reviews = [r for r in reviews if r["user_id"] != cid]

    if len(self_reviews) < 3:
        continue
    self_reviews = sorted(self_reviews, key=lambda x: x["timestamp"])

    supervised_reviews = self_reviews[-3:]
    profile_reviews = self_reviews[:-3] + friend_reviews
    train_records.append({
        "center_user_id": cid,
        "profile": profile_reviews,
        "target": supervised_reviews[0],
        "split": "train"
    })

    val_records.append({
        "center_user_id": cid,
        "profile": profile_reviews,
        "target": supervised_reviews[1],
        "split": "val"
    })
    test_records.append({
        "center_user_id": cid,
        "profile": profile_reviews,
        "target": supervised_reviews[2],
        "split": "test"
    })

print(f"Saving results to {BASE_DIR}...")
with open(TRAIN_PATH, "w", encoding="utf-8") as f:
    json.dump(train_records, f, ensure_ascii=False, indent=2)
with open(VAL_PATH, "w", encoding="utf-8") as f:
    json.dump(val_records, f, ensure_ascii=False, indent=2)
with open(TEST_PATH, "w", encoding="utf-8") as f:
    json.dump(test_records, f, ensure_ascii=False, indent=2)

print("\nDone!")
print(f"Train samples: {len(train_records)}")
print(f"Val samples:   {len(val_records)}")
print(f"Test samples:  {len(test_records)}")
