"""FACT 风格的 frame–action 交叉注意力分割模型（**按机制重写的简化实现**）。

**机制出处**：Lu et al., *FACT: Frame-Action Cross-Attention Temporal Modeling for Efficient
Action Segmentation*, CVPR 2024（[CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Lu_FACT_Frame-Action_Cross-Attention_Temporal_Modeling_for_Efficient_Action_Segmentation_CVPR_2024_paper.html)，MIT 许可）。
论文的关键设计是三件：①并行 frame 分支与 action 分支；②两支之间**双向交叉注意力**；
③**匹配损失**把 action token 与真值动作段绑定，使"一个 token 编码一整段"。
论文消融显示去掉 action-token 损失会把 F1@10 从 79.1 打到 50.1，即**匹配损失是它的价值核心**，
不是可有可无的正则。

**⚠ 实现边界（必须知情）**：本文件是按上述机制**重写**的，不是论文代码的移植——
本项目没有论文全文与官方代码，因此以下细节属本实现的取舍，与论文未必逐字一致：

- token 数 ``num_tokens`` 由超参给出（论文按其数据规模设定；本项目单视频段数少，默认 32）；
- 匹配用标准匈牙利算法（``scipy.optimize.linear_sum_assignment``），代价 =
  分类交叉熵 + 时序代价；论文的具体代价形式未在调研材料中核实；
- 未匹配 token 归入一个额外的 ``no-object`` 类（DETR 式惯例）；
- 逐帧监督仍走流水线的类别加权 CE（为了与本项目其它臂**口径可比**），token 匹配损失是**附加项**。

**为什么值得试**：本项目已实测的两类失败是"动作段被 idle 吞掉"与"段被切碎"
（`asformer h32` 每个真值段内平均产生 2.99 个预测段），而 FACT 是唯一**按构造**要求
"一个 token ↔ 一整段"的架构。

**复杂度**：交叉注意力是 ``O(T·K)``（K 为 token 数），比全注意力 ``O(T²)`` 便宜——本项目最长
训练序列 1635 帧，这是相对 asformer 的显著优势。

**非因果**（token 看整段序列）→ 只能进 ``full_sequence_temporal`` 流水线。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mstcn import _apply_sequence_normalization, _resolve_sequence_normalization


class FrameBranchBlock(nn.Module):
    """frame 分支的膨胀卷积残差块（前置 LayerNorm，[B,T,D] 布局）。"""

    def __init__(self, dim: int, dilation: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(self.norm(x).transpose(1, 2)).transpose(1, 2)
        return x + self.dropout(F.relu(z))


class CrossAttentionBlock(nn.Module):
    """双向交叉注意力块：action token 从帧聚合，帧再从 token 取段级上下文。

    两个方向都是"目标作 query、来源作 key/value"，各自前置 LayerNorm + 残差 + FFN。
    """

    def __init__(self, dim: int, nhead: int, dropout: float):
        super().__init__()
        self.action_norm = nn.LayerNorm(dim)
        self.frame_norm = nn.LayerNorm(dim)
        self.action_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.frame_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.action_ffn = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim * 4, dim),
        )
        self.frame_ffn = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim * 4, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, frames: torch.Tensor):
        """``tokens [B,K,D]``、``frames [B,T,D]``；返回更新后的 ``(tokens, frames)``。"""

        attended, _ = self.action_attn(self.action_norm(tokens), frames, frames, need_weights=False)
        tokens = tokens + self.dropout(attended)
        tokens = tokens + self.dropout(self.action_ffn(tokens))
        attended, _ = self.frame_attn(self.frame_norm(frames), tokens, tokens, need_weights=False)
        frames = frames + self.dropout(attended)
        frames = frames + self.dropout(self.frame_ffn(frames))
        return tokens, frames


def extract_segments(labels) -> list[tuple[int, int, int]]:
    """从逐帧标签 ``[T]`` 提取非 idle 的极大连续段，返回 ``[(start, end, class), ...]``（右开区间）。

    idle 约定为类别 0（本项目 catalog 的类别顺序把 idle 放在首位）。
    接受 ``torch.Tensor`` 或 ``np.ndarray``——诊断脚本常直接传 numpy 标签。
    """

    arr = labels.detach().cpu().numpy() if hasattr(labels, "detach") else np.asarray(labels)
    change = np.concatenate([[0], np.where(np.diff(arr) != 0)[0] + 1, [len(arr)]])
    return [(int(s), int(e), int(arr[s])) for s, e in zip(change[:-1], change[1:]) if arr[s] != 0]


class FACTNet(nn.Module):
    """FACT 风格分割网络：frame 分支 + action token 分支 + 双向交叉注意力 + 段匹配损失。

    ``forward`` 输入 ``[B,T,F]``、输出逐帧 logits ``[B,T,C]``（供推理/评估）；
    训练损失经 ``compute_loss`` 汇总：**类别加权逐帧 CE（与其它臂同口径）+ 段匹配损失**。

    ``num_tokens`` 是 action token 数。每个视频单独做一次一对一匹配，因此该值应不低于
    单视频的真值动作段数（本项目 train 单视频最多约 30 段，默认 32）。
    匹配不上的 token 归入 ``no-object`` 类（类别维为 ``C+1``）。
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 64,
        nhead: int = 4,
        num_blocks: int = 3,
        num_tokens: int = 32,
        frame_layers: int = 3,
        dropout: float = 0.3,
        matching_weight: float = 1.0,
        temporal_weight: float = 1.0,
        no_object_weight: float = 0.1,
        output_mode: str = "frame",
        sequence_normalization: str = "none",
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"FACT d_model 必须能被 nhead 整除: d_model={d_model}, nhead={nhead}")
        if num_tokens < 1:
            raise ValueError(f"num_tokens 必须 ≥1，实际 {num_tokens}")
        self.sequence_normalization = _resolve_sequence_normalization(sequence_normalization)
        self.num_classes = num_classes
        self.num_tokens = num_tokens
        self.matching_weight = matching_weight
        self.temporal_weight = temporal_weight
        # no-object 项降权（DETR 惯例）：未匹配 token 通常远多于匹配 token（本项目 32 token
        # vs 每视频约 7.4 个真值段），等权会让 no-object 项主导并诱发 token 全塌缩到 no-object。
        self.no_object_weight = no_object_weight
        # 输出模式：
        # - "frame"：逐帧头直接输出（token 只经交叉注意力提供"软"上下文）
        # - "token"：把 token 的**时间分布 × 类别分布**混合成逐帧分布（token 直接充当输出）。
        #   后者才是"匹配损失绑定 token"的设计意图——否则逐帧 CE 可以完全无视 token 分支，
        #   实测正是如此：token 分支已训好（no-object 仅 10.3%）而逐帧输出仍碎（段数比 3.98）。
        if output_mode not in {"frame", "token"}:
            raise ValueError(f"output_mode 必须是 'frame' 或 'token'，实际 {output_mode!r}")
        self.output_mode = output_mode

        self.input_projection = nn.Linear(input_dim, d_model)
        self.frame_branch = nn.ModuleList(
            FrameBranchBlock(d_model, dilation=2 ** (i % 4), dropout=dropout)
            for i in range(frame_layers)
        )
        # action token：可学习、跨样本共享，靠交叉注意力从帧里聚合"段"的信息。
        self.tokens = nn.Parameter(torch.zeros(num_tokens, d_model))
        nn.init.normal_(self.tokens, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList(
            CrossAttentionBlock(d_model, nhead, dropout) for _ in range(num_blocks)
        )
        self.frame_head = nn.Linear(d_model, num_classes)
        # token 分类头多一维：no-object（未匹配到任何真值段）
        self.token_class_head = nn.Linear(d_model, num_classes + 1)
        # token 时序分布由「token 表示 × 帧表示」的相似度给出（见 _forward_all），不引入额外参数：
        # 它正是交叉注意力让 token 覆盖某一段的副产品，与 FACT「token 编码一整段」的语义一致。
        self.temporal_scale = d_model ** -0.5
        self.register_buffer("norm_mean", torch.zeros(1, 1, input_dim))
        self.register_buffer("norm_std", torch.ones(1, 1, input_dim))

    def _forward_all(self, x: torch.Tensor):
        """完整前向，返回 ``(frame_logits [B,T,C], token_logits [B,K,C+1], token_time [B,K,T])``。

        ``token_time`` 是**对数**时间分布（沿 T 归一化），用于匹配代价与段覆盖损失。
        """

        x = (x - self.norm_mean) / self.norm_std
        x = _apply_sequence_normalization(x, self.sequence_normalization)
        frames = self.input_projection(x)
        for block in self.frame_branch:
            frames = block(frames)
        tokens = self.tokens.unsqueeze(0).expand(frames.shape[0], -1, -1)
        for block in self.blocks:
            tokens, frames = block(tokens, frames)
        # token 在时间轴上的分布 = token 表示与逐帧表示的缩放点积，再对时间轴做 log_softmax。
        token_time = torch.matmul(tokens, frames.transpose(1, 2)) * self.temporal_scale  # [B,K,T]
        token_time = F.log_softmax(token_time, dim=-1)
        token_logits = self.token_class_head(tokens)
        return self.frame_head(frames), token_logits, token_time

    def _token_mixture_logits(self, token_logits: torch.Tensor, token_time: torch.Tensor):
        """把 token 的**对数时间分布**与**类别对数分布**混合成逐帧对数概率 ``[B,T,C]``。

        ``log P(t, c) = logsumexp_k [ log w_k(t) + log P_k(c) ]``——即"每个 token 是一个
        (段位置, 类别) 的假设"，逐帧输出是这些假设的混合。no-object 维不参与输出。

        **两处归一化缺一不可**（本函数第一版漏了两处，输出概率和只有 0.107）：

        1. ``token_time`` 只沿 **T** 归一化，沿 **k** 的权重之和不为 1 → 需对 k 再做一次
           ``log_softmax``（``log w_k(t)``）；
        2. 类别维截断掉 no-object 后 ``Σ_c P_k(c) < 1`` → 需对截断后的 C 维重新
           ``log_softmax``。

        两个都做，输出才满足 ``Σ_c exp(out[t, c]) = 1``（已由单元测试钉住）。
        """

        logp_class = F.log_softmax(token_logits[..., : self.num_classes], dim=-1)  # 对 C 重新归一化
        logw = F.log_softmax(token_time, dim=1)                                    # 对 k 归一化
        mixed = logw.unsqueeze(-1) + logp_class.unsqueeze(2)                       # [B,K,T,C]
        return torch.logsumexp(mixed, dim=1)                                       # [B,T,C]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 推理约定：逐帧 logits [B,T,C]。
        frame_logits, token_logits, token_time = self._forward_all(x)
        if self.output_mode == "token":
            return self._token_mixture_logits(token_logits, token_time)
        return frame_logits

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        """逐帧加权 CE + 段匹配损失（Hungarian）。

        ``criterion`` 是流水线拥有的**类别加权 CE**（监督口径随数据走），逐帧项直接用它，
        以保证与 `mstcn2` / `asformer` / `actionness_tcn` 各臂**口径可比**；
        token 匹配损失是本架构的附加项（论文的价值核心）。
        """

        frame_logits, token_logits, token_time = self._forward_all(x)
        # 逐帧项始终监督**推理实际使用的输出**（token 模式下即混合分布），保证口径一致。
        used = (self._token_mixture_logits(token_logits, token_time)
                if self.output_mode == "token" else frame_logits)
        loss = criterion(used.reshape(-1, self.num_classes), y.reshape(-1))
        loss = loss + self.matching_weight * self._matching_loss(token_logits, token_time, y)
        return loss

    def _matching_loss(self, token_logits, token_time, y) -> torch.Tensor:
        """对每个样本做一次 action token ↔ 真值段的一对一匈牙利匹配，再算匹配损失。

        代价 = 分类交叉熵（token 对真值类）+ ``temporal_weight`` × 时序代价
        （该 token 落在真值段内的概率缺口 ``1 − Σ_{t∈段} P(t)``）。
        匹配上的 token 用真值类、未匹配的用 no-object 类各算一次交叉熵。
        """

        from scipy.optimize import linear_sum_assignment

        no_object = self.num_classes
        total = token_logits.new_zeros(())
        batch = token_logits.shape[0]
        for b in range(batch):
            segments = extract_segments(y[b])
            logp_class = F.log_softmax(token_logits[b], dim=-1)   # [K, C+1]
            prob_time = token_time[b].exp()                       # [K, T]
            if not segments:
                # 该样本没有任何动作段：全部 token 都是 no-object。
                total = total + self.no_object_weight * F.cross_entropy(
                    token_logits[b], torch.full((self.num_tokens,), no_object,
                                                dtype=torch.long, device=token_logits.device)
                )
                continue
            cost = np.zeros((len(segments), self.num_tokens), dtype=np.float64)
            logp_np = logp_class.detach().cpu().numpy()
            prob_np = prob_time.detach().cpu().numpy()
            for i, (s, e, cls) in enumerate(segments):
                cost[i] = -logp_np[:, cls] + self.temporal_weight * (1.0 - prob_np[:, s:e].sum(axis=1))
            rows, cols = linear_sum_assignment(cost)
            matched = torch.as_tensor(cols, dtype=torch.long, device=token_logits.device)
            labels = torch.as_tensor([segments[r][2] for r in rows],
                                     dtype=torch.long, device=token_logits.device)
            total = total + F.cross_entropy(token_logits[b][matched], labels)
            # 时序项：让匹配上的 token 尽可能覆盖它对应的整段（右开区间）。
            index = torch.arange(token_time.shape[-1], device=token_time.device)
            temp = token_time.new_zeros(())
            for (s, e, _cls), col in zip((segments[r] for r in rows), cols):
                inside = (index >= s) & (index < e)
                temp = temp - torch.logsumexp(token_time[b, col][inside], dim=0)
            total = total + self.temporal_weight * temp / max(len(rows), 1)
            unmatched = torch.ones(self.num_tokens, dtype=torch.bool, device=token_logits.device)
            unmatched[matched] = False
            if unmatched.any():
                total = total + self.no_object_weight * F.cross_entropy(
                    token_logits[b][unmatched],
                    torch.full((int(unmatched.sum()),), no_object,
                               dtype=torch.long, device=token_logits.device),
                )
        return total / max(batch, 1)

    def normalizer_spec(self) -> str:
        """归一化口径的溯源声明（本模型恒按训练集 z-score 归一化）。"""

        return ("zscore/train-set/buffers/v1" if self.sequence_normalization == "none"
                else f"zscore/train-set/buffers/v1+sequence-{self.sequence_normalization}/v1")

    def fit_normalization(self, features: list) -> None:
        """训练前钩子：按训练集 z-score 统计写入归一化 buffer（口径同其它全序列模型）。"""

        arr = np.concatenate(features, axis=0)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        std[std < 1e-4] = 1.0
        dev = self.norm_mean.device
        self.norm_mean.copy_(torch.tensor(mean, dtype=torch.float32, device=dev).reshape(1, 1, -1))
        self.norm_std.copy_(torch.tensor(std, dtype=torch.float32, device=dev).reshape(1, 1, -1))
