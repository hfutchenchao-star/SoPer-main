import os
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import to_undirected

class SocialGraphDataset(Dataset):

    def __init__(self, npy_path, emb_root, use_degree_feat=True, transform=None):
        super().__init__(None, transform, None)
        self.graph_dict = np.load(npy_path, allow_pickle=True).item()
        self.keys = list(self.graph_dict.keys())
        self.use_degree_feat = use_degree_feat
        self.emb_root = emb_root  

    def len(self):
        return len(self.keys)

    def get(self, idx):
        center_uid = self.keys[idx]
        g = self.graph_dict[center_uid]
        nodes = g["nodes"]
        edges = g["edges"]

        nid = {u: i for i, u in enumerate(nodes)}
        edge_list = [[nid[u], nid[v]] for u, v in edges if u in nid and v in nid]

        if len(edge_list) == 0:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_index = to_undirected(edge_index)

        N = len(nodes)
        deg = torch.bincount(edge_index[0], minlength=N).float().unsqueeze(1)
        deg_norm = deg / (deg.max() + 1e-6) if self.use_degree_feat else torch.ones_like(deg)

        embs = []
        for uid in nodes:
            emb_path = os.path.join(self.emb_root, center_uid, f"{uid}.emb")
            if not os.path.exists(emb_path):
                raise FileNotFoundError(f"找不到节点 {uid} 的 embedding 文件: {emb_path}")

            try:
                try:
                    tensor_data = torch.load(emb_path, map_location="cpu")
                    if isinstance(tensor_data, torch.Tensor):
                        arr = tensor_data.numpy()
                    elif isinstance(tensor_data, list):
                        arr = torch.stack(tensor_data).numpy()
                    else:
                        raise ValueError(f"未知embedding类型: {type(tensor_data)}")
                except Exception:
                    arr = np.loadtxt(emb_path)

                if arr.ndim == 1:
                    arr = arr[np.newaxis, :]
                mean_emb = arr.mean(axis=0)

            except Exception as e:
                raise RuntimeError(f"加载 {emb_path} 时出错: {e}")

            embs.append(mean_emb)

        embs = torch.tensor(np.stack(embs), dtype=torch.float32)

        x = torch.cat([deg_norm, embs], dim=1)

        center_local = nid[center_uid]
        data = Data(x=x, edge_index=edge_index)
        data.center_uid = center_uid
        data.center_local = torch.tensor(center_local, dtype=torch.long)
        data.node_names = nodes

        return data
