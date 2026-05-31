"""Qwen3.5 模型架构源码（带完整中文注释）

原始来源: HuggingFace Transformers - modeling_qwen3_5.py
GitHub: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py

Qwen3.5 是 2026 年初发布的最新一代 Qwen 模型，相比 Qwen2 有革命性的架构变化。

====================================================================
Qwen3.5 架构总览 —— 混合注意力架构（核心创新）
====================================================================

Qwen3.5 最大的创新：交替使用 Softmax 注意力和线性注意力

    输入 token_ids
         ↓
    Token Embedding
         ↓
    × N 层 Qwen3_5DecoderLayer ────────────────────────────┐
    ┌─────────────────────────────────────────────────┐    │
    │  Qwen3_5RMSNorm (Pre-Norm, 新公式: (1+w)*x)     │    │
    │       ↓                                          │    │
    │  ┌─ 根据 layer_type 分派 ──────────────────┐    │    │
    │  │                                          │    │    │
    │  │  "full_attention" (每4层1次):            │    │    │
    │  │    Qwen3_5Attention                      │    │    │
    │  │    ├── Q/K RMSNorm（新增！）              │    │    │
    │  │    ├── RoPE（仅 25% 维度，partial=0.25）  │    │    │
    │  │    ├── 标准 softmax(QK^T/√d)V            │    │    │
    │  │    └── Sigmoid 门控输出（新增！）         │    │    │
    │  │                                          │    │    │
    │  │  "linear_attention" (其余3/4层):         │    │    │
    │  │    Qwen3_5GatedDeltaNet（全新组件！）     │    │    │
    │  │    ├── Causal Conv1D（局部上下文）         │    │    │
    │  │    ├── 可学习衰减参数 A_log, dt_bias      │    │    │
    │  │    ├── Gated Delta Rule（线性递归）        │    │    │
    │  │    └── RMSNormGated 输出归一化            │    │    │
    │  └──────────────────────────────────────────┘    │    │
    │       ↓                                          │    │
    │  残差连接 (+)                                     │    │
    │       ↓                                          │    │
    │  Qwen3_5RMSNorm                                  │    │
    │       ↓                                          │    │
    │  Qwen3_5MLP (SwiGLU FFN，与 Qwen2 相同)          │    │
    │       ↓                                          │    │
    │  残差连接 (+)                                     │    │
    └─────────────────────────────────────────────────┘    │
         ↓                                                  │
    Final Qwen3_5RMSNorm                                    │
         ↓                                                  │
    LM Head → logits                                        │

多模态版本额外包含:
    - Vision Encoder: 3D Patch Embedding + ViT Blocks + Patch Merger
    - MRoPE: 3D 旋转位置编码 (时间/高度/宽度)
    - Image/Video Token 嵌入注入

====================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass


# ===========================================================================
# 模型配置
# ===========================================================================

@dataclass
class Qwen3_5TextConfig:
    """Qwen3.5 文本模型配置（精简版）

    默认参数对应 Qwen3.5-9B-Instruct。

    相比 Qwen2 的关键新增配置:
    - head_dim=256 (Qwen2 中是 hidden_size/num_heads 自动计算)
    - attention_bias=False (Qwen2 中 Q/K/V 有 bias=True)
    - partial_rotary_factor=0.25 (Qwen2 中 RoPE 覆盖全部维度)
    - linear_* 系列参数: 线性注意力的配置
    - layer_types: 混合注意力层的类型分配

    默认 layer_types 模式 (full_attention_interval=4):
    [linear, linear, linear, full, linear, linear, linear, full, ...]
    即每 4 层中有 1 层用标准 softmax 注意力，3 层用线性注意力
    """
    # 基础参数
    vocab_size: int = 248320                 # 词表大小（比 Qwen2 的 152064 大）
    hidden_size: int = 4096                  # 隐藏维度
    intermediate_size: int = 12288           # FFN 中间维度
    num_hidden_layers: int = 32              # Decoder 层数
    num_attention_heads: int = 16            # Q 头数
    num_key_value_heads: int = 4             # K/V 头数（GQA）
    head_dim: int = 256                      # 每头维度（显式配置，而非自动计算）
    hidden_act: str = "silu"                 # FFN 激活
    max_position_embeddings: int = 32768     # 最大序列长度
    rms_norm_eps: float = 1e-6              # RMSNorm epsilon
    attention_dropout: float = 0.0           # 注意力 dropout
    attention_bias: bool = False             # 注意力投影是否有 bias（Qwen2 为 True）
    tie_word_embeddings: bool = False        # 是否共享 Embedding/LM Head

    # RoPE 参数
    rope_theta: float = 1000000.0            # RoPE 基础频率
    partial_rotary_factor: float = 0.25      # 【新增】仅 25% 的 head_dim 应用 RoPE

    # 线性注意力参数（Qwen3.5 核心新增）
    linear_conv_kernel_dim: int = 4          # Causal Conv1D 的卷积核大小
    linear_key_head_dim: int = 128           # 线性注意力 K 头维度
    linear_value_head_dim: int = 128         # 线性注意力 V 头维度
    linear_num_key_heads: int = 16           # 线性注意力 K 头数
    linear_num_value_heads: int = 32         # 线性注意力 V 头数

    # 层类型（混合注意力架构）
    layer_types: Optional[list[str]] = None  # 每层的注意力类型
    full_attention_interval: int = 4         # 全注意力层的间隔（默认每4层1次full）

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = [
                "linear_attention" if bool((i + 1) % self.full_attention_interval) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]


# ===========================================================================
# Qwen3_5RMSNorm — 新版 RMSNorm（公式变化）
# ===========================================================================

class Qwen3_5RMSNorm(nn.Module):
    """Qwen3.5 的 RMSNorm（公式与 Qwen2 不同！）

    Qwen2:  output = x * rsqrt(mean(x²) + eps) * weight       (weight 初始化为 1)
    Qwen3.5: output = x * rsqrt(mean(x²) + eps) * (1 + weight) (weight 初始化为 0)

    关键区别:
    - Qwen3.5 使用 (1 + weight)，weight 初始化为 0
      → 初始时 output = x * rsqrt(mean(x²) + eps) * 1 = 纯 RMSNorm（无缩放）
      → 训练过程中 weight 可以学出正值（增强）或负值（抑制）
    - Qwen2 使用 weight，初始化为 1
      → 初始时 output = x * rsqrt(mean(x²) + eps) * 1 = 同样是纯 RMSNorm

    两者初始行为相同，但 Qwen3.5 的参数化更稳定:
    - weight=0 是更自然的初始化点
    - 梯度更新时不会因为 weight 偏离 1 太远而导致数值问题

    参考: https://github.com/huggingface/transformers/pull/29402
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))  # 初始化为 0（Qwen2 初始化为 1）

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())  # (1 + w) 而非 w
        return output.type_as(x)


# ===========================================================================
# Qwen3_5RMSNormGated — 带门控的 RMSNorm（用于 GatedDeltaNet）
# ===========================================================================

class Qwen3_5RMSNormGated(nn.Module):
    """带 SiLU 门控的 RMSNorm（GatedDeltaNet 专用）

    output = RMSNorm(hidden_states) * SiLU(gate)

    将归一化和门控结合在一起，用于线性注意力的输出处理。
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        # 门控: 用 gate 信号调制归一化后的输出
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


# ===========================================================================
# MRoPE — 多维旋转位置编码（新增！）
# ===========================================================================

class Qwen3_5TextRotaryEmbedding(nn.Module):
    """Qwen3.5 的多维旋转位置编码 (MRoPE)

    相比 Qwen2 的标准 RoPE，MRoPE 支持 3D 位置编码:
    - 维度 0: Temporal（时间/序列位置）
    - 维度 1: Height（图像/视频高度位置）
    - 维度 2: Width（图像/视频宽度位置）

    对于纯文本: 三个维度使用相同的位置序列
    对于图像/视频: 各维度使用独立的 2D/3D 位置

    interleaved layout:
    将频率从 [TTT...HHH...WWW] 重排为 [THWTHWTHW...TT]
    使得相邻维度覆盖不同的位置维度，增强位置信息的表达能力。

    partial_rotary_factor=0.25:
    只有 25% 的 head_dim 维度应用 RoPE，其余保持不变。
    这使得大部分维度专注于内容建模，小部分负责位置感知。
    """

    def __init__(self, config: Qwen3_5TextConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.config = config

        # 计算逆频率
        base = config.rope_theta
        dim = int(config.head_dim * config.partial_rotary_factor)  # 只对 25% 维度应用 RoPE
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0
        self.mrope_section = [11, 11, 10]  # MRoPE 的 3D 维度分段

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """将 3D 频率从 [TTT...HHH...WWW] 重排为交错 [THWTHWTHW...TT]

        交错布局让每个注意力头的不同维度对同时编码不同的位置维度，
        比分块布局提供更丰富的位置信息。
        """
        freqs_t = freqs[0]  # T 维度作为基础
        for dim, offset in enumerate((1, 2), start=1):  # H, W 维度
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    @torch.no_grad()
    def forward(self, x, position_ids):
        """
        Args:
            x: 用于获取设备/dtype 的张量
            position_ids: 纯文本时 (batch, seq_len)，多模态时 (3, batch, seq_len)

        Returns:
            cos: (batch, seq_len, head_dim) 余弦编码
            sin: (batch, seq_len, head_dim) 正弦编码
        """
        # 纯文本: 扩展为 3D (3, batch, seq_len)
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        inv_freq_expanded = (
            self.inv_freq[None, None, :, None].float()
            .expand(3, position_ids.shape[1], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, :, None, :].float()

        # 计算 3D 频率
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        # 应用交错 MRoPE 布局
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ===========================================================================
# rotate_half / apply_rotary_pos_emb — 支持 partial rotary
# ===========================================================================

def rotate_half(x):
    """旋转操作（与 Qwen2 相同）"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """应用旋转位置编码（支持 partial rotary）

    与 Qwen2 的区别:
    Qwen2:   对 Q/K 的全部维度应用 RoPE
    Qwen3.5: 只对前 rotary_dim 个维度应用 RoPE，其余保持不变
             → 部分维度编码位置，部分维度专注于内容
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]

    # 分离: 前 rotary_dim 维度 → 应用 RoPE; 后面的维度 → 保持不变
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # 拼接回完整维度
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


def repeat_kv(hidden_states, n_rep):
    """GQA: 复制 K/V 头（与 Qwen2 完全相同）"""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# ===========================================================================
# Qwen3_5Attention — 标准 Softmax 注意力（增强版）
# ===========================================================================

class Qwen3_5Attention(nn.Module):
    """Qwen3.5 的标准 Softmax 注意力层

    相比 Qwen2 Attention 的 3 个重要变化:

    1. Q/K 归一化 (q_norm / k_norm):
       在注意力计算前对 Q 和 K 的每个头做 RMSNorm。
       → 稳定训练，防止 Q/K 范数爆炸导致注意力坍缩
       → 这是 Qwen3 从 OLMo 系列借鉴的技术

    2. Query 门控:
       q_proj 输出维度是 2 × (n_heads × head_dim)，拆分为 query 和 gate。
       output = sigmoid(gate) * attention_output
       → 门控机制让模型可以选择性忽略某些注意力模式

    3. Partial RoPE (25%):
       只有 head_dim 的 25% 应用旋转位置编码
       → 更多维度专注于内容建模

    投影层:
    - q_proj: (hidden_size → n_heads × head_dim × 2)  ← 2x 因为要拆出 gate
    - k_proj: (hidden_size → n_kv_heads × head_dim)
    - v_proj: (hidden_size → n_kv_heads × head_dim)
    - o_proj: (n_heads × head_dim → hidden_size)
    - q_norm: RMSNorm(head_dim)  ← 新增
    - k_norm: RMSNorm(head_dim)  ← 新增
    """

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        # Q 投影: 2x 维度（一半是 query，一半是 gate）
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim * 2, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        # 新增: Q/K 头级别归一化
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[dict] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """前向传播

        流程:
        1. Q 投影 → 拆分为 query 和 gate
        2. Q/K 归一化 (RMSNorm)
        3. 应用 RoPE (partial, 25%)
        4. KV Cache 更新
        5. GQA 复制 K/V
        6. Softmax 注意力计算
        7. Sigmoid(gate) * output → 门控调制
        8. 输出投影
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # ---- Step 1: Q 投影 + 拆分 query/gate ----
        qkv_output = self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2)
        query_states, gate = torch.chunk(qkv_output, 2, dim=-1)  # 拆分！
        gate = gate.reshape(*input_shape, -1)

        # ---- Step 2: Q/K 归一化 ----
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # ---- Step 3: Partial RoPE ----
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # ---- Step 4: KV Cache ----
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        # ---- Step 5: GQA ----
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # ---- Step 6: Softmax 注意力 ----
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        if self.training and self.attention_dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)

        # ---- Step 7: 门控调制 ----
        attn_output = attn_output * torch.sigmoid(gate)  # 【新增！】sigmoid 门控

        # ---- Step 8: 输出投影 ----
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# ===========================================================================
# Qwen3_5GatedDeltaNet — 线性注意力层（全新组件！核心创新）
# ===========================================================================

class Qwen3_5GatedDeltaNet(nn.Module):
    """Gated Delta Network — Qwen3.5 的线性注意力层

    这是 Qwen3.5 最核心的架构创新。
    替代标准 Softmax 注意力，使用线性注意力 + 递归状态更新。

    为什么需要线性注意力?
    - Softmax 注意力的复杂度: O(n²)（序列长度的平方）
    - 线性注意力的复杂度: O(n)（线性增长）
    - 对于长序列（100K+ tokens），Softmax 注意力太慢太费显存
    - 线性注意力牺牲少量精度，换取巨大的效率提升

    Gated Delta Rule 核心思想:
    1. 维护一个递归状态 S_t（类似 RNN 的隐状态）
    2. 每步: 计算当前 K/V 与状态的差异（Delta）
    3. 用门控 β 控制更新幅度
    4. 用衰减 g 控制历史信息的保留程度
    5. 输出 = S_t × Q（从状态中读取信息）

    架构组件:
    - Causal Conv1D: 捕获局部上下文（弥补线性注意力对局部信息的不足）
    - in_proj_qkv: 投影 Q, K, V
    - in_proj_z: 门控信号 z
    - in_proj_b: 更新强度 β (sigmoid)
    - in_proj_a: 衰减率 g (通过 A_log 和 dt_bias 参数化)
    - RMSNormGated: 归一化 + 门控输出

    推理模式:
    - Prefill (多 token): 使用 chunk 模式（并行计算）
    - Decode (单 token): 使用 recurrent 模式（逐步更新状态）
    → 类似于 KV Cache，但维护的是固定大小的递归状态，而非线性增长的缓存

    参数说明:
    - A_log: 控制衰减速率，A 越大 → 衰减越快 → 更关注近期信息
    - dt_bias: 时间步偏置，微调每个头的衰减行为
    - conv_kernel_size=4: Conv1D 的窗口大小（4 个 token 的局部上下文）
    """

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads     # 线性注意力的 V 头数
        self.num_k_heads = config.linear_num_key_heads       # 线性注意力的 K 头数
        self.head_k_dim = config.linear_key_head_dim         # K 头维度
        self.head_v_dim = config.linear_value_head_dim       # V 头维度
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.act = F.silu

        # 输入投影
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)    # 门控信号
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)  # 更新强度
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)  # 衰减率

        # Causal Conv1D: 捕获局部上下文
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim, out_channels=self.conv_dim,
            bias=False, kernel_size=self.conv_kernel_size,
            groups=self.conv_dim, padding=self.conv_kernel_size - 1,
        )

        # 可学习的时间步参数
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))  # log 空间参数化，确保 A > 0

        # 归一化 + 门控输出
        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        流程:
        1. 投影 QKV / z / b / a
        2. Causal Conv1D（局部上下文）
        3. 计算衰减 g 和更新强度 β
        4. GQA: 复制 K 头以匹配 V 头
        5. Gated Delta Rule: 更新递归状态 + 计算输出
        6. RMSNormGated 归一化 + 门控
        7. 输出投影
        """
        batch_size, seq_len, _ = hidden_states.shape

        # ---- Step 1: 投影 ----
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)  # (B, conv_dim, seq)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)  # 更新强度
        a = self.in_proj_a(hidden_states)  # 衰减率

        # ---- Step 2: Causal Conv1D ----
        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])
        mixed_qkv = mixed_qkv.transpose(1, 2)

        # ---- Step 3: 拆分 Q, K, V ----
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        # ---- Step 4: 计算衰减和更新强度 ----
        beta = b.sigmoid()                                  # 更新强度: (0, 1)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)  # 衰减率: 负值

        # ---- Step 5: GQA (K/V 头数可能不同) ----
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        # ---- Step 6: Gated Delta Rule 核心计算 ----
        # (简化版 — 完整实现有 chunk 和 recurrent 两种模式)
        # 这里展示 chunk 模式的核心逻辑:
        scale = 1 / (query.shape[-1] ** 0.5)
        query_scaled = query * scale

        # 递归状态初始化
        state = torch.zeros(batch_size, self.num_v_heads, self.head_k_dim, self.head_v_dim,
                           dtype=value.dtype, device=value.device)
        output = torch.zeros(batch_size, self.num_v_heads, seq_len, self.head_v_dim,
                            dtype=value.dtype, device=value.device)

        for t in range(seq_len):
            q_t = query_scaled[:, :, t]       # (B, n_heads, d_k)
            k_t = key[:, :, t]                # (B, n_heads, d_k)
            v_t = value[:, :, t]              # (B, n_heads, d_v)
            g_t = g[:, :, t].exp().unsqueeze(-1).unsqueeze(-1)  # 衰减因子
            beta_t = beta[:, :, t].unsqueeze(-1)                 # 更新强度

            # 递归更新: state = g_t * state + k_t ⊗ (beta_t * (v_t - state × k_t))
            state = state * g_t
            kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)     # 从状态中读取
            delta = (v_t - kv_mem) * beta_t                      # 计算更新量
            state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            output[:, :, t] = (state * q_t.unsqueeze(-1)).sum(dim=-2)  # 输出

        output = output.transpose(1, 2).contiguous()  # (B, seq, n_heads, d_v)

        # ---- Step 7: RMSNormGated ----
        output = output.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        output = self.norm(output, z)
        output = output.reshape(batch_size, seq_len, -1)

        return self.out_proj(output)


# ===========================================================================
# Qwen3_5MLP — SwiGLU FFN（与 Qwen2 相同）
# ===========================================================================

class Qwen3_5MLP(nn.Module):
    """SwiGLU 前馈网络（与 Qwen2 完全相同）

    output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
    """

    def __init__(self, config: Qwen3_5TextConfig, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.act_fn = F.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ===========================================================================
# Qwen3_5DecoderLayer — 混合注意力解码器层
# ===========================================================================

class Qwen3_5DecoderLayer(nn.Module):
    """Qwen3.5 解码器层（混合注意力架构的核心）

    与 Qwen2 的关键区别:
    - Qwen2: 所有层都使用标准 Softmax 注意力
    - Qwen3.5: 根据 layer_type 分派到不同注意力实现
      - "full_attention": Qwen3_5Attention (标准 softmax，带 Q/K norm + 门控)
      - "linear_attention": Qwen3_5GatedDeltaNet (线性注意力，O(n) 复杂度)

    默认模式 (full_attention_interval=4):
    Layer 0: linear_attention
    Layer 1: linear_attention
    Layer 2: linear_attention
    Layer 3: full_attention  ← 每 4 层一次标准注意力
    Layer 4: linear_attention
    ...

    这意味着 75% 的层使用线性注意力（快），25% 使用标准注意力（准）。
    线性注意力层负责高效的序列建模，标准注意力层确保高精度的全局信息聚合。
    """

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]

        # 根据层类型选择注意力实现
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)

        self.mlp = Qwen3_5MLP(config, config.intermediate_size)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Token Mixer: 根据层类型分派
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                **kwargs,
            )
        elif self.layer_type == "full_attention":
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = residual + hidden_states

        # FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
