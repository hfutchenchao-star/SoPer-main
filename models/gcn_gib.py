import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# --- 可复用：RBF核与HSIC（如不需要，可删除） ---
def kernel_matrix(x, sigma):
    gram = torch.mm(x, x.t())
    return torch.exp((gram - 1) / sigma)

def hsic(Kx, Ky, m):
    Kxy = torch.mm(Kx, Ky)
    h = torch.trace(Kxy) / (m ** 2) + Kx.mean() * Ky.mean() - 2 * Kxy.mean() / m
    return h * (m / (m - 1)) ** 2

# --- 边掩码 MLP（保持不变，供 graph_learner 使用） ---
class EdgeMaskMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.mlp(x).squeeze(-1)

class SocialGCN_GBSR(nn.Module):
    """
    结构级 GIB（GBSR 风格）版：
      - 先用 graph_learner 基于节点嵌入为每条社交边打“可靠性”分数 -> edge_mask ∈ [0,1]
      - 所有 GCN 层共享同一组 edge_weight（结构级瓶颈），强调社交关系可靠性
      - 支持 PyG 多子图 batch（edge_index 为拼接图索引）
      - 保留 LLM projector、中心节点提取、（可选）HSIC
    """
    def __init__(
        self,
        in_channels=1,
        hidden_channels=64,
        out_channels=64,
        num_layers=3,
        dropout=0.1,
        projector_dim=256,
        # GIB/采样相关
        temperature=0.2,
        edge_bias=0.0,
        detach_mask=False,      # 是否像原 GBSR 一样在构图后 detach 掉 mask（默认 False，端到端可导）
        # HSIC 可选
        gib_sigma=0.5,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.temperature = temperature
        self.edge_bias = edge_bias
        self.detach_mask = detach_mask
        self.gib_sigma = gib_sigma

        # 输入投影到GCN隐藏维
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # GCN 层（各层共享同一 edge_weight -> 结构级掩码）
        self.gcn_convs = nn.ModuleList([
            GCNConv(hidden_channels if i < num_layers-1 else hidden_channels,
                    hidden_channels if i < num_layers-1 else out_channels,
                    add_self_loops=False, normalize=False)
            for i in range(num_layers)
        ])

        # GBSR 风格的 graph_learner（两层 MLP）
        self.activate = nn.ReLU()
        self.linear_1 = nn.Linear(2 * hidden_channels, hidden_channels, bias=True)
        self.linear_2 = nn.Linear(hidden_channels, 1, bias=True)

        # LLM Projector（保留你的接口）
        self.llm_projector = nn.Sequential(
            nn.Linear(out_channels, projector_dim),
            nn.LayerNorm(projector_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projector_dim, projector_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ---------- 结构级 graph learner（GBSR 核心） ----------
    def graph_learner(self, node_emb, edge_index, is_training=True):
        """
        基于节点嵌入, 为每条边生成可靠性 edge_mask ∈ [0,1]
        node_emb: [N, H]  (已投影到 hidden_channels 的节点表征)
        edge_index: [2, E]
        """
        row, col = edge_index
        cat = torch.cat([node_emb[row], node_emb[col]], dim=1)  # [E, 2H]
        h = self.activate(self.linear_1(cat))
        logit = self.linear_2(h).view(-1)                      # [E]

        if is_training:
            # Gumbel-Sigmoid（同 GBSR 思路）
            eps = torch.rand_like(logit).clamp_(1e-8, 1-1e-8)
            gumbel = -torch.log(-torch.log(eps))
            s = (logit + gumbel) / self.temperature
            mask = torch.sigmoid(s) + self.edge_bias
        else:
            mask = torch.sigmoid(logit) + self.edge_bias

        # 是否 detach（对齐原 GBSR 的“仅构图不反传”做法）
        if self.detach_mask:
            mask = mask.detach()

        return mask  # [E]

    def forward(self, x, edge_index, batch_idx=None, return_all=False):
        """
        结构级 GIB
        """
        # (1) 先得到 emb_before（输入投影后）
        x = x.to(next(self.input_proj.parameters()).dtype)
        emb_before = self.input_proj(x)  # [N, H]
        emb_before = F.dropout(emb_before, p=self.dropout, training=self.training)

        # (2) graph_learner 生成结构级边掩码
        edge_mask = self.graph_learner(emb_before, edge_index, is_training=self.training)

        # (3) 多层 GCN 传播（共享 edge_mask）
        all_emb = [emb_before]
        h = emb_before
        for li in range(self.num_layers):
            h = self.gcn_convs[li](h, edge_index, edge_weight=edge_mask)
            if li < self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            all_emb.append(h)
        emb_after = torch.stack(all_emb, dim=1).mean(dim=1)

        if not return_all:
            return emb_after

        # (4) 同时返回 emb_before、emb_after、edge_mask
        return {
            "emb_before": emb_before,
            "emb_after": emb_after,
            "edge_mask": edge_mask
        }

    # ---------- 可选：与原接口兼容的中心节点提取 ----------
    def extract_center_embeddings(self, batch_data):
        """
        提取 batch 中每个图的中心节点 embedding（使用结构级掩码）
        需要 batch_data[i].center_local
        """
        result = self.forward(batch_data.x, batch_data.edge_index, batch_data.batch, return_all=True)
        emb = result["emb"]
        # PyG Batch 索引辅助
        ptr = batch_data.ptr  # [B+1]
        B = batch_data.num_graphs

        centers = []
        for i in range(B):
            center_local = batch_data[i].center_local.item()
            global_idx = ptr[i] + center_local
            centers.append(emb[global_idx])
        centers = torch.stack(centers)  # [B, out_channels]
        return centers, result["edge_mask"]  # 返回中心 embedding + 边可靠性

    def get_llm_prompts(self, batch_data):
        """
        中心节点 -> projector -> LLM prompt 向量
        """
        centers, edge_mask = self.extract_center_embeddings(batch_data)
        prompt = self.llm_projector(centers)
        return prompt, {"edge_mask": edge_mask}

    # ---------- 可选：HSIC（若你还想加信息瓶颈的正则项） ----------
    def compute_hsic_loss(self, emb_before, emb_after):
        """
        若你想用 HSIC 约束“掩码前后”的独立性/信息量，可在外部传入两路表征做正则。
        这里保留接口，按需使用。
        """
        m = emb_before.size(0)
        if m < 2:
            return torch.tensor(0.0, device=emb_before.device)
        e1 = F.normalize(emb_before, p=2, dim=1)
        e2 = F.normalize(emb_after,  p=2, dim=1)
        Kx = kernel_matrix(e1, self.gib_sigma)
        Ky = kernel_matrix(e2, self.gib_sigma)
        return hsic(Kx, Ky, m)
