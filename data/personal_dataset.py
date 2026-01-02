import torch
import datasets
import numpy as np
from tqdm import tqdm
from utils.templates import Qwen2PromptTemplate
from data.social_datasets import SocialGraphDataset
import os
import sys


class PersonalDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            main_dataset,
            user_his_emb_map,  # {center_user_id: {user_id: Tensor[num_reviews, D]}}
            llm_tokenizer,
            max_length=3072,
            max_friend_len=8,
            new_tokens=None,
            training=True,
    ):
        self.main_dataset = main_dataset
        self.llm_tokenizer = llm_tokenizer
        self.total_len = len(main_dataset)
        self.training = training
        self.user_his_emb_map = user_his_emb_map
        self.max_friend_len = max_friend_len
        # 你可以把这个路径提前定义在脚本最上面
        SOCIAL_GRAPH_PATH = "/root/autodl-tmp/DEP-main/datasets/social_graphs.npy"

        if not os.path.exists(SOCIAL_GRAPH_PATH):
            raise FileNotFoundError(f" Graph file not found: {SOCIAL_GRAPH_PATH}")

        # 只加载一次（建议放到函数外面全局加载，这里是示例）
        social_graphs = np.load(SOCIAL_GRAPH_PATH, allow_pickle=True).item()
        social_dataset = SocialGraphDataset(SOCIAL_GRAPH_PATH)

        # 冷启动任务 system prompt
        system_prompt = (
            f"Given the user's social graph information, the user's one past review, and several past reviews from the user's friends "
            f"(each review including review stars, review business categories, review text, review embeddings and each friend's review including a weight), "
            f"as well as the output review stars and output review business categories, "
            f"generate a personalized business review text for the user. You should consider these weights, where higher-weight friends' reviews have a greater influence on the user's preferences when generating the review text.\n"
            f"Note: [Social Graph Embedding] denotes a soft prompt embedding of the user's social structure, and [Review Embedding] denotes a soft prompt embedding of the review text. "
            f"[Social Graph Embedding] and [Review Embedding] should serve as semantic hints to guide the generation of personalized business reviews.\n"
        )

        self.pt = Qwen2PromptTemplate(system_prompt)
        self.processed_data = []

        for idx in tqdm(range(self.total_len), desc="Building cold-start dataset"):
            sample = self.main_dataset[idx]
            center_user_id = sample["center_user_id"]
            profile = sample["profile"]
            target = sample["target"]

            categories = target["categories"]
            stars = target["stars"]
            out_str = target["text"]

            # === Step 1. 区分用户评论与好友评论 ===
            user_reviews = [p for p in profile if p["user_id"] == center_user_id]
            friend_reviews = [p for p in profile if p["user_id"] != center_user_id]

            # 只取最近一条用户评论
            user_reviews = sorted(user_reviews, key=lambda x: x.get("date", ""))
            user_review = user_reviews[-1] if len(user_reviews) > 0 else None

            # 取时间最近的若干好友评论
            friend_reviews = sorted(friend_reviews, key=lambda x: x.get("date", ""), reverse=True)
            friends_selected = friend_reviews[:self.max_friend_len]

            # === Step 2. 用户评论 embedding ===
            if user_review is not None:
                uid = user_review["user_id"]
                emb = None

                # 检查该中心用户的 embedding 是否存在
                if center_user_id not in self.user_his_emb_map:
                    raise KeyError(f"[ERROR] Missing embedding map for center user: {center_user_id}")

                # 检查该用户（可能是中心用户自己）是否在中心用户的 embedding 子图中
                if uid not in self.user_his_emb_map[center_user_id]:
                    raise KeyError(f"[ERROR] Missing embedding for user {uid} under center user {center_user_id}")

                all_embs = self.user_his_emb_map[center_user_id][uid]  # Tensor[num_reviews, D]

                # 评论按时间升序，因此找到该评论对应的索引位置
                if user_review not in user_reviews:
                    raise ValueError(f"[ERROR] user_review not found in user_reviews for {uid}")

                review_idx = user_reviews.index(user_review)
                if review_idx >= all_embs.shape[0]:
                    raise IndexError(
                        f"[ERROR] review_idx {review_idx} exceeds embedding size {all_embs.shape[0]} "
                        f"for user {uid} under center {center_user_id}"
                    )

                emb = all_embs[review_idx]
                user_emb = emb
            else:
                raise ValueError(f"[ERROR] user_review is None for center user {center_user_id}")

            # === Step 3. 固定长度好友槽位（按社交图顺序） ===

            # 1) 构造 {fid: 该好友的评论(按时间新->旧)}
            friends_by_id = {}
            for rec in profile:
                if rec["user_id"] != center_user_id:
                    friends_by_id.setdefault(rec["user_id"], []).append(rec)
            for fid in friends_by_id:
                friends_by_id[fid] = sorted(friends_by_id[fid], key=lambda x: x.get("date", ""), reverse=True)

            # 2) 从社交图文件中提取好友顺序
            # ---------------------------------------------------------

            if center_user_id in social_graphs:
                graph_info = social_graphs[center_user_id]
                node_order = graph_info["nodes"]  # 原始节点顺序
                # 去掉中心用户自己，只保留好友
                graph_friend_order = [n for n in node_order if n != center_user_id]
            else:
                # 如果找不到该用户图，退化为默认字母排序
                graph_friend_order = sorted(friends_by_id.keys())

            # 3) 过滤掉没有评论的好友（避免空）
            sorted_fids = [fid for fid in graph_friend_order if fid in friends_by_id]

            # 2) 选前 K 个好友；如果好友不足 K，则 K' = 实际好友数
            K = min(self.max_friend_len, len(sorted_fids)) if len(sorted_fids) > 0 else 0
            primary_fids = sorted_fids[:K]

            if K == 0:
                raise ValueError(f"[ERROR] No friends found for center user {center_user_id}")

            # 3) 检查 embedding 映射完整性，并准备各好友的所有 embs（升序与生成顺序一致）
            #    注意：我们用于 token 的评论选“最近的一条”，即 friends_by_id[fid][0]，其 embedding 索引来自升序表尾
            friend_all_embs = {}  # fid -> Tensor[num_reviews, D]（按时间升序）
            friend_mean_embs = {}  # fid -> Tensor[1, D]

            for fid in primary_fids:
                if center_user_id not in self.user_his_emb_map or fid not in self.user_his_emb_map[center_user_id]:
                    raise KeyError(f"[ERROR] Missing emb for friend {fid} under center {center_user_id}")
                all_embs = self.user_his_emb_map[center_user_id][fid]
                if not hasattr(all_embs, "ndim") or all_embs.ndim != 2:
                    raise ValueError(
                        f"[ERROR] Embedding for friend {fid} must be 2D, got {getattr(all_embs, 'shape', None)}")
                friend_all_embs[fid] = all_embs
                friend_mean_embs[fid] = all_embs.mean(dim=0, keepdim=True)

            # 4) 先为每个好友取“最近的一条评论”（friends_by_id 是新->旧）
            slots = []  # 长度将填充到 self.max_friend_len；元素是 (fid, review_obj, emb_tensor)
            for fid in primary_fids:
                reviews_desc = friends_by_id[fid]  # 新->旧
                # 找到这条“最近评论”在升序索引中的位置：升序列表的最后一个元素
                reviews_asc = sorted(reviews_desc, key=lambda x: x.get("date", ""))
                latest_review = reviews_desc[0]
                # 升序中索引
                review_idx_asc = reviews_asc.index(latest_review)
                # 对齐 emb
                all_embs = friend_all_embs[fid]
                if review_idx_asc >= all_embs.shape[0]:
                    raise IndexError(
                        f"[ERROR] friend {fid} emb count {all_embs.shape[0]} < needed index {review_idx_asc}"
                    )
                slots.append((fid, latest_review, all_embs[review_idx_asc]))

            # 5) 若好友数 < 固定槽位数：按“好友轮询 + 各自第2近/第3近…”补齐
            round_idx = 1  # 第2近开始
            while len(slots) < self.max_friend_len:
                filled_in_this_round = False
                for fid in primary_fids:
                    reviews_desc = friends_by_id[fid]
                    if round_idx < len(reviews_desc):
                        r = reviews_desc[round_idx]  # 第 round_idx+1 近
                        # 找到升序索引并取 emb
                        reviews_asc = sorted(reviews_desc, key=lambda x: x.get("date", ""))
                        review_idx_asc = reviews_asc.index(r)
                        all_embs = friend_all_embs[fid]
                        if review_idx_asc >= all_embs.shape[0]:
                            raise IndexError(
                                f"[ERROR] friend {fid} emb count {all_embs.shape[0]} < idx {review_idx_asc}"
                            )
                        slots.append((fid, r, all_embs[review_idx_asc]))
                        filled_in_this_round = True
                        if len(slots) >= self.max_friend_len:
                            break
                if not filled_in_this_round:
                    # 已无更多评论可补，允许重复最近的一条，确保长度固定
                    last_fid, last_r, last_e = slots[-1]
                    while len(slots) < self.max_friend_len:
                        slots.append((last_fid, last_r, last_e))
                    break
                round_idx += 1

            # === Step 4. 构造输入字符串 ===
            inp_str = (
                "[Social Graph Embedding]: "
                "<social_graph_token_start>[SOCIAL_GRAPH_TOKEN]<social_graph_token_end>\n"
            )

            # ========== 用户评论部分 ==========
            if user_review:
                user_text = clean_text(user_review["text"])
                inp_str += "[User's Review]:\n"
                inp_str += (
                    f"- [Review Stars]: {user_review['stars']}\n"
                    f"- [Review Business Categories]: {', '.join(user_review.get('categories', []))}\n"
                    f"- [Review Text]: {user_text}\n"
                    f"- [Review Embedding]: <user_token_start>[USER_TOKEN_0]<user_token_end>\n"
                )
            else:
                inp_str += "[User's Review]: None\n"

            # ========== 好友评论部分（单独构建 friend_message） ==========
            inp_str = "[Reviews from User’s Friends]:\n"
            for j in range(self.max_friend_len):
                fid, fr, _ = slots[j]
                friend_text = clean_text(fr["text"])
                inp_str += (
                    f"- [User's Friend Review {j + 1}]:\n"
                    f"  - [Review Stars]: {fr['stars']}\n"
                    f"  - [Review Business Categories]: {', '.join(fr.get('categories', []))}\n"
                    f"  - [Review Text]: {friend_text}\n"
                    f"  - [Weight]: 1.0\n"
                    f"  - [Review Embedding]: <friend_token_start>[FRIEND_TOKEN_{j}]<friend_token_end>\n"
                )

            # ========== 输出目标信息 ==========
            inp_str += f"[Output Review Stars]: {stars}\n"
            inp_str += f"[Output Review Business Categories]: {', '.join(categories)}\n"

            # ========== 调用 Prompt 模板 ==========
            inp_str = self.pt.build_prompt(user_message=inp_str)

            # === Step 5. Tokenize ===
            total_max_length = max_length + 2048 + 1
            inputs = self.llm_tokenizer(
                inp_str,
                add_special_tokens=False,
                truncation=False  # ❗ 禁止截断，否则你看不到真实长度
            )
            input_len = len(inputs["input_ids"])
            if input_len > max_length:
                # ✅ 抛出异常（或改成 print / log.warning）
                print(
                    f"⚠️ 跳过超长样本：当前 {input_len} > 上限 {max_length} "
                    f"({input_len - max_length} tokens 超出)"
                )
                continue  # ✅ 跳过当前循环，处理下一个样本

            targets = self.llm_tokenizer(
                out_str,
                max_length=max_length,
                truncation=True,
                add_special_tokens=False
            )

            # === Step 6. 拼接选中评论 embedding ===
            all_embs_tensor = [user_emb]  # 用户 emb
            for j in range(self.max_friend_len):
                fid, fr, emb = slots[j]
                all_embs_tensor.append(emb)
            all_embs_tensor = torch.stack(all_embs_tensor, dim=0)

            if center_user_id in social_dataset.keys:
                ind = social_dataset.keys.index(center_user_id)
                graph_data = social_dataset.get(ind)  # ✅ 这是 PyG Data
            else:
                raise ValueError(
                    f" 未找到中心用户 {center_user_id} 的社交图数据！请检查 social_graphs.npy 是否包含该用户。")

            out_str = clean_text(out_str)

            data = {
                "inp_str": inp_str,
                "out_str": out_str,
                "center_user_id": center_user_id,
                "embeddings": all_embs_tensor.tolist(),
                "graph_data": {
                    "x": graph_data.x.tolist(),
                    "edge_index": graph_data.edge_index.tolist(),
                    "center_local": graph_data.center_local.item(),
                    "center_uid": graph_data.center_uid,
                }
            }
            print(inp_str)
            print(out_str)

            self.processed_data.append(data)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        return self.processed_data[idx]


def convert_to_dataset(dataset):
    def gen():
        for data in dataset:
            yield data

    return datasets.Dataset.from_generator(gen)


def clean_text(text: str) -> str:
    """去除换行、首尾空格、多余空格"""
    if not text:
        return ""
    # 替换换行和制表符为空格
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    # 连续空格压缩成一个
    text = " ".join(text.split())
    return text.strip()