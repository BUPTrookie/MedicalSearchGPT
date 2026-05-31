"""Qwen2 / Qwen2.5 模型架构源码（带完整中文注释）

原始来源: HuggingFace Transformers - modeling_qwen2.py
GitHub: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py

注意: Qwen2 和 Qwen2.5 在 HuggingFace Transformers 中共用同一份模型代码，
区别仅在配置（权重、超参数），不影响架构本身。

本文件是原始源码的精简注释版，移除了 HuggingFace 框架耦合部分
（GenerationMixin, PreTrainedModel, 缓存系统等），
只保留核心架构组件，便于理解模型原理。

====================================================================
Qwen2 架构总览
====================================================================

Qwen2 是一个 Decoder-Only Transformer，架构与 LLaMA 高度相似：

    输入 token_ids
         ↓
    Token Embedding (nn.Embedding)
         ↓
    × N 层 Qwen2DecoderLayer ─────────────────────────┐
    ┌─────────────────────────────────────────────┐    │
    │  RMSNorm (Pre-Norm)                          │    │
    │       ↓                                      │    │
    │  Qwen2Attention (GQA + RoPE + KV Cache)      │    │
    │       ↓                                      │    │
    │  残差连接                                     │    │
    │       ↓                                      │    │
    │  RMSNorm (Pre-Norm)                          │    │
    │       ↓                                      │    │
    │  Qwen2MLP (SwiGLU FFN)                       │    │
    │       ↓                                      │    │
    │  残差连接                                     │    │
    └─────────────────────────────────────────────┘    │
         ↓                                              │
    Final RMSNorm                                       │
         ↓                                              │
    LM Head (Linear) → logits                          │
         ↓                                              │
    CrossEntropy Loss / 自回归生成                      │

与 LLaMA 的主要区别:
1. Q/K/V 投影使用 bias=True（LLaMA 为 bias=False）
2. 支持 sliding window attention（部分层使用局部注意力）
3. 支持 layer_types 配置（不同层可使用不同注意力策略）

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
class Qwen2Config:
    """Qwen2 模型配置（精简版，仅保留架构相关参数）

    以 Qwen2.5-7B-Instruct 为例的典型参数:
        hidden_size=3584, num_hidden_layers=28, num_attention_heads=28,
        num_key_value_heads=4, intermediate_size=18944, max_position_embeddings=131072

    以 Qwen2.5-0.5B 为例:
        hidden_size=896, num_hidden_layers=24, num_attention_heads=14,
        num_key_value_heads=2, intermediate_size=4864
    """
    hidden_size: int = 3584                 # 隐藏维度 d_model
    num_hidden_layers: int = 28             # Decoder 层数
    num_attention_heads: int = 28           # Q 头数
    num_key_value_heads: int = 4            # K/V 头数（GQA，远小于 Q 头数）
    intermediate_size: int = 18944          # FFN 中间维度
    max_position_embeddings: int = 131072   # 最大序列长度（128K）
    vocab_size: int = 152064                # 词表大小
    rms_norm_eps: float = 1e-6              # RMSNorm epsilon
    rope_theta: float = 1000000.0           # RoPE 基础频率（比 LLaMA 的 10000 大 100 倍，支持更长上下文）
    hidden_act: str = "silu"                # FFN 激活函数
    attention_dropout: float = 0.0          # 注意力 dropout
    tie_word_embeddings: bool = False       # 是否共享 Embedding 和 LM Head（Qwen2.5 不共享）
    sliding_window: int = 131072            # 滑动窗口大小（局部注意力的窗口）
    pad_token_id: int = 151643              # padding token id

    @property
    def head_dim(self) -> int:
        """每个注意力头的维度"""
        return self.hidden_size // self.num_attention_heads

    @property
    def rope_parameters(self) -> dict:
        """RoPE 参数（兼容原始 HuggingFace 接口）"""
        return {"rope_type": "default", "rope_theta": self.rope_theta}

    @property
    def layer_types(self) -> list[str]:
        """每层的注意力类型
        Qwen2 可以配置不同层使用不同注意力策略:
        - "full_attention": 全局注意力（标准）
        - "sliding_attention": 滑动窗口注意力（只看最近 W 个 token，省显存）
        """
        return ["full_attention"] * self.num_hidden_layers


# ===========================================================================
# Qwen2MLP — SwiGLU 前馈网络
# ===========================================================================

class Qwen2MLP(nn.Module):
    """Qwen2 的前馈网络，使用 SwiGLU 激活

    结构:
        output = down_proj( act_fn(gate_proj(x)) * up_proj(x) )

    三个线性层:
    - gate_proj: (hidden_size → intermediate_size) — 主投影，经过激活函数
    - up_proj:   (hidden_size → intermediate_size) — 门控投影，不经过激活
    - down_proj: (intermediate_size → hidden_size) — 下投影回原始维度

    SwiGLU = SiLU(W_gate(x)) * W_up(x)
    其中 SiLU(x) = x * sigmoid(x)，也称为 Swish 激活函数

    与标准 ReLU FFN 的区别:
    - 多一个 up_proj（门控），参数量增加约 50%
    - 但效果显著优于 ReLU/GeLU，现代 LLM 的标配
    """

    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # gate_proj: 主投影路径，经过激活函数
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # up_proj: 门控路径，与激活后的 gate 相乘
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # down_proj: 将中间维度投影回隐藏维度
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        # 激活函数，Qwen2 使用 SiLU (Swish)
        # SiLU(x) = x * sigmoid(x)，比 ReLU 更平滑，梯度更稳定
        self.act_fn = F.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, hidden_size)
        → gate_proj(x):  (batch, seq_len, intermediate_size) → SiLU 激活
        → up_proj(x):    (batch, seq_len, intermediate_size) → 门控信号
        → 逐元素相乘 → down_proj → (batch, seq_len, hidden_size)
        """
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


# ===========================================================================
# Qwen2RotaryEmbedding — 旋转位置编码
# ===========================================================================

class Qwen2RotaryEmbedding(nn.Module):
    """Qwen2 的旋转位置编码 (RoPE)

    RoPE 的核心思想:
    通过旋转矩阵将绝对位置信息编码到 Q 和 K 中，
    使得 Q·K^T 的点积自然包含相对位置信息。

    频率公式: inv_freq[i] = 1 / (θ^(2i/d_head))
    其中 θ = rope_theta (Qwen2 用 1000000，比 LLaMA 的 10000 大 100 倍)

    θ 越大 → 频率衰减越慢 → 远程位置信息保留越好 → 支持更长上下文

    前向传播:
    1. 根据 position_ids 计算每个位置的频率: freqs = inv_freq × positions
    2. 将频率转为 cos/sin 值
    3. cos/sin 用于旋转 Q 和 K 的向量
    """

    def __init__(self, config: Qwen2Config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.config = config

        # 计算逆频率: inv_freq[i] = 1 / (theta^(2i/d_head))
        # head_dim 维度被分成 d_head/2 组频率，每组覆盖一对维度
        dim = config.head_dim
        base = config.rope_theta  # Qwen2: 1000000.0
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0  # 后处理缩放因子，默认 RoPE 类型不使用

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor):
        """计算给定位置的 cos/sin 编码

        Args:
            x:            用于获取设备和数据类型的张量
            position_ids: (batch, seq_len) 每个位置的绝对位置索引

        Returns:
            cos: (batch, seq_len, head_dim) 余弦编码
            sin: (batch, seq_len, head_dim) 正弦编码
        """
        # inv_freq: (head_dim/2,) → 扩展为 (batch, head_dim/2, 1)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        # position_ids: (batch, seq_len) → (batch, 1, seq_len)
        position_ids_expanded = position_ids[:, None, :].float()

        # 矩阵乘法: (batch, head_dim/2, 1) @ (batch, 1, seq_len) = (batch, head_dim/2, seq_len)
        # 转置: (batch, seq_len, head_dim/2)
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)

        # 拼接: (batch, seq_len, head_dim/2) → (batch, seq_len, head_dim)
        # 前 half 和后 half 使用相同频率，因为 rotate_half 会交换前后半部分
        emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """旋转操作: 将向量后半部分取负后与前半部分交换

    这是 RoPE 的核心操作，等价于复数域的旋转变换。
    将 d_head 维向量视为 d_head/2 个复数对，
    每对 (x1, x2) 旋转为 (-x2, x1)。

    例: [a, b, c, d] → [-c, -d, a, b]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """对 Q 和 K 应用旋转位置编码

    公式: q_rot = q * cos(θ) + rotate_half(q) * sin(θ)

    等价于将 q 视为复数向量后乘以 e^{iθ}（旋转 θ 角度）。
    由于 cos 和 sin 只与绝对位置有关，
    而 Q·K^T 内积后自然包含相对位置差 (m-n) 的信息。

    Args:
        q:             (batch, n_heads, seq_len, head_dim)
        k:             (batch, n_kv_heads, seq_len, head_dim)
        cos:           (batch, seq_len, head_dim)
        sin:           (batch, seq_len, head_dim)
        unsqueeze_dim: 在哪个维度 unsqueeze cos/sin 以匹配 q/k 的形状

    Returns:
        (q_rotated, k_rotated) 编码了位置信息的 Q 和 K
    """
    cos = cos.unsqueeze(unsqueeze_dim)  # (batch, 1, seq_len, head_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """将 K/V 头复制 n_rep 次以匹配 Q 头数（GQA 核心操作）

    Grouped-Query Attention (GQA):
    当 K/V 头数 < Q 头数时，每个 K/V 头被多个 Q 头共享。
    这减少了 KV Cache 的显存占用（训练和推理都受益）。

    例: Qwen2.5-7B: 28 Q heads, 4 KV heads → n_rep=7, 节省 85% KV Cache
    例: Qwen2.5-0.5B: 14 Q heads, 2 KV heads → n_rep=7

    形状变化:
    输入:  (batch, n_kv_heads, seq_len, head_dim)
    输出:  (batch, n_kv_heads * n_rep, seq_len, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states  # MHA（标准多头注意力），无需复制
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# ===========================================================================
# Qwen2Attention — 多头自注意力（支持 GQA）
# ===========================================================================

class Qwen2Attention(nn.Module):
    """Qwen2 多头自注意力

    核心特性:
    1. Grouped-Query Attention (GQA): K/V 头数远小于 Q 头数
       → 减少推理时 KV Cache 显存，加速生成
    2. RoPE 旋转位置编码: 相对位置编码，支持长上下文外推
    3. KV Cache: 缓存历史 K/V，避免重复计算
    4. Sliding Window Attention: 可选，部分层只看局部窗口

    与 LLaMA Attention 的区别:
    - Qwen2 的 Q/K/V 投影使用 bias=True（LLaMA 为 bias=False）
    - Qwen2 支持 sliding_window（滑动窗口注意力）
    - Qwen2 使用更激进的 GQA（KV heads 更少）

    线性投影:
    - q_proj: (hidden_size → num_heads * head_dim)        Q 投影
    - k_proj: (hidden_size → num_kv_heads * head_dim)     K 投影（GQA: 更小）
    - v_proj: (hidden_size → num_kv_heads * head_dim)     V 投影（GQA: 更小）
    - o_proj: (num_heads * head_dim → hidden_size)        输出投影
    """

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim

        # GQA: Q 头数 / K/V 头数 = 每组复制的次数
        # 例: 28 heads / 4 kv_heads = 7 (每个 KV 头被 7 个 Q 头共享)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads

        # 注意力缩放因子: 1/sqrt(d_head)
        self.scaling = self.head_dim ** -0.5

        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        # Q/K/V 投影（注意: Qwen2 使用 bias=True，与 LLaMA 不同）
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)

        # 输出投影（无偏置）
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        # 滑动窗口（如果该层配置为 sliding_attention）
        layer_type = config.layer_types[layer_idx] if layer_idx < len(config.layer_types) else "full_attention"
        self.sliding_window = config.sliding_window if layer_type == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,                          # (batch, seq_len, hidden_size)
        position_embeddings: tuple[torch.Tensor, torch.Tensor], # (cos, sin) RoPE 编码
        attention_mask: Optional[torch.Tensor] = None,         # 因果掩码
        past_key_values: Optional[dict] = None,                # KV Cache
        cache_position: Optional[torch.LongTensor] = None,     # 缓存位置索引
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """前向传播

        流程:
        1. 线性投影 Q, K, V
        2. 应用 RoPE 旋转位置编码
        3. 更新 KV Cache（如果有）
        4. GQA: 复制 K/V 头
        5. 计算注意力: softmax(QK^T / √d) V
        6. 输出投影

        Returns:
            (attn_output, attn_weights)
        """
        input_shape = hidden_states.shape[:-1]  # (batch, seq_len)
        hidden_shape = (*input_shape, -1, self.head_dim)

        # ---- Step 1: 线性投影 + reshape 为多头格式 ----
        # (batch, seq_len, hidden_size) → (batch, seq_len, n_heads, head_dim) → (batch, n_heads, seq_len, head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # ---- Step 2: 应用 RoPE ----
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # ---- Step 3: KV Cache 更新 ----
        if past_key_values is not None:
            # 将新的 K/V 拼接到缓存中，返回完整的 K/V
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        # ---- Step 4: GQA — 复制 K/V 头以匹配 Q 头数 ----
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # ---- Step 5: 计算注意力 ----
        # QK^T / √d
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        # 应用因果掩码（上三角为 -inf → softmax 后为 0 → 看不到未来位置）
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax（用 float32 计算以保证数值精度）
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Dropout（训练时）
        if self.training and self.attention_dropout > 0:
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # 加权求和
        attn_output = torch.matmul(attn_weights, value_states)

        # ---- Step 6: 合并多头 + 输出投影 ----
        # (batch, n_heads, seq_len, head_dim) → (batch, seq_len, n_heads * head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


# ===========================================================================
# Qwen2RMSNorm — RMS 归一化
# ===========================================================================

class Qwen2RMSNorm(nn.Module):
    """RMS 归一化层（与 LLaMA 相同）

    与 LayerNorm 的区别:
    - 不计算均值（只计算均方根），更高效
    - 没有偏置参数（只有缩放参数 weight）
    - 在 float32 下计算以保证精度

    公式: output = x / sqrt(mean(x²) + eps) * weight
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # 可学习缩放参数
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        # 在 float32 下计算以确保数值稳定
        hidden_states = hidden_states.to(torch.float32)
        # 计算方差（均方）
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # rsqrt = 1/sqrt，比先 sqrt 再除更高效
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# ===========================================================================
# Qwen2DecoderLayer — 解码器层
# ===========================================================================

class Qwen2DecoderLayer(nn.Module):
    """Qwen2 解码器层（Pre-Norm 结构）

    结构:
        residual = x
        x = RMSNorm(x)
        x = Attention(x) + residual     ← 注意力子层 + 残差连接

        residual = x
        x = RMSNorm(x)
        x = MLP(x) + residual           ← FFN 子层 + 残差连接

    Pre-Norm vs Post-Norm:
    - Pre-Norm（先归一化再计算子层）: 训练更稳定，现代 LLM 标配
    - Post-Norm（先计算子层再归一化）: 原始 Transformer，训练需要 warmup
    """

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # 自注意力层
        self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx)

        # SwiGLU FFN
        self.mlp = Qwen2MLP(config)

        # 两个 RMSNorm: 注意力前 和 FFN 前
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 该层的注意力类型（full / sliding）
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[dict] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states:       (batch, seq_len, hidden_size)
            attention_mask:      因果掩码
            position_ids:        位置索引
            past_key_values:     KV Cache
            use_cache:           是否使用缓存
            cache_position:      缓存位置
            position_embeddings: RoPE 的 (cos, sin) 编码

        Returns:
            (batch, seq_len, hidden_size)
        """
        # ---- 注意力子层 ----
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)  # Pre-Norm

        # 自注意力
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states  # 残差连接

        # ---- FFN 子层 ----
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)  # Pre-Norm
        hidden_states = self.mlp(hidden_states)                       # SwiGLU FFN
        hidden_states = residual + hidden_states                       # 残差连接

        return hidden_states


# ===========================================================================
# 输出数据结构
# ===========================================================================

@dataclass
class BaseModelOutputWithPast:
    """模型基础输出"""
    last_hidden_state: torch.Tensor      # (batch, seq_len, hidden_size)
    past_key_values: Optional[dict] = None  # KV Cache（用于生成）


@dataclass
class CausalLMOutputWithPast:
    """因果语言模型输出"""
    loss: Optional[torch.Tensor] = None          # 交叉熵损失
    logits: Optional[torch.Tensor] = None         # (batch, seq_len, vocab_size)
    past_key_values: Optional[dict] = None        # KV Cache
    hidden_states: Optional[tuple] = None         # 各层隐状态


# ===========================================================================
# Qwen2Model — 模型主体（Embedding + Decoder Layers + Final Norm）
# ===========================================================================

class Qwen2Model(nn.Module):
    """Qwen2 模型主体

    包含:
    - Token Embedding: token id → 隐向量
    - N 层 Qwen2DecoderLayer
    - Final RMSNorm
    - Rotary Embedding（全局共享）

    不包含 LM Head（由 Qwen2ForCausalLM 添加）。
    """

    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Token 嵌入层: token id → hidden_size 维向量
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # N 层解码器
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        # 最终 RMSNorm（所有层输出后归一化）
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 旋转位置编码（全局共享，所有层使用同一组频率）
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        self.has_sliding_layers = "sliding_attention" in config.layer_types

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """创建因果掩码（下三角矩阵）

        确保位置 i 只能看到 ≤ i 的位置信息（自回归性质）
        上三角为 -inf，softmax 后为 0（看不到未来）
        """
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1
        )
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: bool = False,
    ) -> BaseModelOutputWithPast:
        """前向传播

        流程:
        1. Token Embedding (或使用预计算的 inputs_embeds)
        2. 构建 RoPE 位置编码
        3. 逐层通过 Decoder Layer
        4. Final RMSNorm

        Args:
            input_ids:      (batch, seq_len) token id 序列
            attention_mask: (batch, seq_len) 注意力掩码（padding mask）
            position_ids:   (batch, seq_len) 位置索引（默认为 0, 1, 2, ...）
            past_key_values: KV Cache（生成时使用）
            inputs_embeds:  预计算的 embedding（与 input_ids 二选一）
            use_cache:      是否返回 KV Cache

        Returns:
            BaseModelOutputWithPast
        """
        # ---- 输入处理 ----
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_len, _ = inputs_embeds.shape

        # 默认位置索引: [0, 1, 2, ..., seq_len-1]
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0)

        # ---- 因果掩码 ----
        if attention_mask is None:
            attention_mask = self._create_causal_mask(seq_len, inputs_embeds.device)

        # ---- KV Cache 初始化 ----
        if use_cache and past_key_values is None:
            past_key_values = [{} for _ in range(self.config.num_hidden_layers)]

        # ---- RoPE 位置编码 ----
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        # ---- 逐层通过 Decoder ----
        hidden_states = inputs_embeds
        for i, decoder_layer in enumerate(self.layers):
            layer_cache = past_key_values[i] if past_key_values is not None else None
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=layer_cache,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
            )

        # ---- 最终归一化 ----
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


# ===========================================================================
# Qwen2ForCausalLM — 因果语言模型（完整可训练模型）
# ===========================================================================

class Qwen2ForCausalLM(nn.Module):
    """Qwen2 因果语言模型（用于文本生成）

    在 Qwen2Model 基础上添加:
    - LM Head: hidden_size → vocab_size 的线性投影
    - Loss 计算: 交叉熵损失（训练时）
    - 生成功能: 自回归逐 token 生成

    权重共享:
    - Qwen2.5 不共享 Embedding 和 LM Head（tie_word_embeddings=False）
    - LLaMA 默认共享

    训练:
    loss = CrossEntropy(logits[:, :-1], labels[:, 1:])
    即用前 n-1 个位置预测后 n-1 个位置（教师强制/Teacher Forcing）
    """

    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size

        # LM Head: 隐状态 → 词表分布
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 权重共享（Qwen2.5 默认不共享）
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
    ) -> CausalLMOutputWithPast:
        """前向传播

        Args:
            input_ids:      (batch, seq_len) 输入 token id
            labels:         (batch, seq_len) 目标 token id（训练时提供）
                            通常 labels = input_ids（shift 由 loss 函数处理）

        Returns:
            CausalLMOutputWithPast，包含 loss（如果有 labels）和 logits
        """
        # ---- 模型前向 ----
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )
        hidden_states = outputs.last_hidden_state

        # ---- LM Head: 隐状态 → logits ----
        logits = self.lm_head(hidden_states)  # (batch, seq_len, vocab_size)

        # ---- 计算损失 ----
        loss = None
        if labels is not None:
            # Shift: 用 logits[:, :-1] 预测 labels[:, 1:]
            # 即每个位置预测下一个 token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # 展平后计算交叉熵
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """自回归生成

        流程:
        1. Prefill: 一次性处理整个 prompt，初始化 KV Cache
        2. Decode: 逐 token 生成
           - 取最后一个位置的 logits
           - 温度缩放 → Top-K/Top-P 过滤 → 采样
           - 将新 token 加入输入序列
           - 利用 KV Cache 只计算新 token

        Args:
            input_ids:       (batch, prompt_len) prompt
            max_new_tokens:  最大生成 token 数
            temperature:     采样温度（越低越确定）
            top_k:           Top-K 采样的 K
            top_p:           Nucleus 采样的累积概率阈值
            eos_token_id:    停止 token

        Returns:
            (batch, prompt_len + generated_len) 完整序列
        """
        self.eval()
        batch_size = input_ids.shape[0]
        generated = input_ids

        # 初始化 KV Cache
        past_key_values = [{} for _ in range(self.config.num_hidden_layers)]

        # ---- Prefill ----
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        causal_mask = self.model._create_causal_mask(seq_len, input_ids.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        next_logits = logits[:, -1, :]  # (batch, vocab_size)

        for step in range(max_new_tokens):
            # 温度缩放
            logits = next_logits / max(temperature, 1e-8)

            # Top-K
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-P (Nucleus)
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove_mask = cum_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[remove_mask] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            # 采样
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)
            generated = torch.cat([generated, next_token], dim=1)

            # EOS 检查
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            # Decode step（只处理新 token）
            new_pos = seq_len + step
            position_ids = torch.tensor([[new_pos]], device=input_ids.device)
            outputs = self.model(
                input_ids=next_token,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = self.lm_head(outputs.last_hidden_state)
            next_logits = logits[:, -1, :]

        return generated
