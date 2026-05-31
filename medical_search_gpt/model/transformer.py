"""Decoder-Only Transformer 完整实现（现代生产版）

架构对标 LLaMA / Qwen2 系列，包含以下现代组件：

┌─────────────────────────────────────────────────────┐
│                    输入 token ids                     │
│                        ↓                              │
│              Token Embedding (查表)                    │
│                        ↓                              │
│              × N 层 DecoderBlock ─────────────────┐   │
│  ┌───────────────────────────────────────────┐    │   │
│  │         RMSNorm (替代 LayerNorm)           │    │   │
│  │                ↓                           │    │   │
│  │  Multi-Head Attention + RoPE + KV Cache    │    │   │
│  │  (支持 GQA 分组查询注意力)                   │    │   │
│  │                ↓                           │    │   │
│  │         残差连接 (+)                        │    │   │
│  │                ↓                           │    │   │
│  │         RMSNorm                            │    │   │
│  │                ↓                           │    │   │
│  │    SwiGLU FFN (w1·silu + w3 门控)          │    │   │
│  │                ↓                           │    │   │
│  │         残差连接 (+)                        │    │   │
│  └───────────────────────────────────────────┘    │   │
│                        ↓                              │
│              Final RMSNorm                            │
│                        ↓                              │
│         LM Head → logits (vocab_size)                 │
│                        ↓                              │
│           Softmax → 采样 → 下一个 token               │
└─────────────────────────────────────────────────────┘

各组件说明：
- RMSNorm:     比 LayerNorm 更高效，不需要计算均值，只计算均方根
- RoPE:        旋转位置编码，通过旋转矩阵编码相对位置，天然支持外推
- GQA:         分组查询注意力，K/V 头数 < Q 头数，减少 KV Cache 显存
- SwiGLU:      SiLU 激活 + 门控线性单元，比 ReLU/GeLU 效果更好
- KV Cache:    缓存已计算的 Key/Value，避免重复计算，加速自回归生成
- Flash Attn:  利用 PyTorch SDPA 的 IO 感知注意力，大幅减少显存和加速
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import ModelConfig


# ===========================================================================
# RMSNorm — Root Mean Square Layer Normalization
# ===========================================================================

class RMSNorm(nn.Module):
    """RMS 归一化层

    与 LayerNorm 的区别：
    - 不计算均值，只计算均方根（RMS）
    - 没有偏置项（bias=False）
    - 计算量更小，效果相当

    公式: output = x / sqrt(mean(x²) + eps) * weight

    Args:
        dim: 归一化的最后一个维度大小（通常为 d_model）
        eps: 防止除零的小常数
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的缩放参数 γ

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            归一化后的张量，形状不变
        """
        # 计算均方根: sqrt(mean(x², dim=-1))
        # 使用 float32 计算以保证数值精度，最后转回原 dtype
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).type_as(x) * self.weight


# ===========================================================================
# RoPE — Rotary Position Embeddings（旋转位置编码）
# ===========================================================================

class RotaryEmbedding(nn.Module):
    """旋转位置编码 (RoPE)

    核心思想：
    将位置信息通过旋转矩阵编码到 Q 和 K 中，使得 Q·K^T 的点积
    自然包含相对位置信息（m-n），无需显式的位置偏置。

    数学原理：
    对于位置 m 的向量 x，旋转角度 θ_m = m * θ_base
    将 x 的相邻维度组成复数对，乘以 e^{iθ_m}（旋转）

    频率序列: inv_freq[i] = 1 / (θ_base^(2i/d_head))
    其中 d_head 是每个注意力头的维度

    Args:
        head_dim:   每个注意力头的维度
        max_seq_len: 最大支持序列长度（预计算缓存）
        theta:      基础频率 θ，LLaMA 用 10000，Qwen2.5 用 1000000
    """

    def __init__(self, head_dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        # inv_freq: (head_dim/2,) 旋转角度的基础频率
        # 每个维度对应不同的频率，低维度频率高（变化快），高维度频率低（变化慢）
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """预计算 cos/sin 缓存表，避免重复计算

        cos_cached: (max_seq_len, head_dim)
        sin_cached: (max_seq_len, head_dim)
        每行对应该位置的旋转角度的 cos/sin 值
        """
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        # freqs: (seq_len, head_dim/2) 每个位置每个频率维度的角度
        freqs = torch.outer(t, self.inv_freq)
        # 拼接为完整维度: (seq_len, head_dim)
        # 前 half 是 cos 部分的频率，后 half 是 sin 部分
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """获取前 seq_len 个位置的 cos/sin 缓存

        Returns:
            cos: (seq_len, head_dim)
            sin: (seq_len, head_dim)
        """
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """将张量的后半部分取负后与前半部分交换

    这是旋转操作的一半：将 [x1, x2] 变为 [-x2, x1]
    配合 cos/sin 实现等价于复数旋转的操作。

    例: [a, b, c, d] → [-c, -d, a, b]
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """对 Q 和 K 应用旋转位置编码

    数学: q_rot = q * cos(θ) + rotate_half(q) * sin(θ)

    这等价于将 q 视为复数向量，乘以 e^{iθ}（旋转 θ 角度）
    由于 cos/sin 只与绝对位置有关，而 Q·K^T 的内积只与相对位置差有关，
    因此天然编码了相对位置信息。

    Args:
        q:   (batch, n_heads, seq_len, head_dim)  查询矩阵
        k:   (batch, n_kv_heads, seq_len, head_dim) 键矩阵
        cos: (seq_len, head_dim) 余弦缓存
        sin: (seq_len, head_dim) 正弦缓存

    Returns:
        (q_rotated, k_rotated) 旋转后的 Q 和 K
    """
    # 广播: (seq_len, head_dim) → (1, 1, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


# ===========================================================================
# KV Cache — 键值缓存（加速自回归生成）
# ===========================================================================

class KVCache:
    """自回归生成的 KV 缓存

    原理：
    在自回归（逐 token）生成时，前面 token 的 K 和 V 已经计算过了。
    缓存它们可以避免重复计算，将每步的计算量从 O(seq_len²) 降到 O(seq_len)。

    流程：
    - Prefill 阶段: 一次性处理全部 prompt，缓存所有 K/V
    - Decode 阶段: 只计算新 token 的 K/V，拼接到缓存中

    内存占用:
    KV Cache 大小 = 2 × n_layers × batch × seq_len × d_model × dtype_size
    例: LLaMA-7B, fp16, 4096 长度 ≈ 2×32×1×4096×4096×2 = 2GB
    """

    __slots__ = ("k_cache", "v_cache")

    def __init__(self):
        self.k_cache: Optional[torch.Tensor] = None  # (batch, n_kv_heads, cached_len, head_dim)
        self.v_cache: Optional[torch.Tensor] = None

    def update(
        self, k: torch.Tensor, v: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """将新的 K/V 拼接到缓存中

        Args:
            k:         (batch, n_kv_heads, new_len, head_dim) 当前步的 Key
            v:         (batch, n_kv_heads, new_len, head_dim) 当前步的 Value
            layer_idx: 当前层索引（预留用于 per-layer cache，当前实现共享）

        Returns:
            (full_k, full_v): 包含历史+当前的完整 K/V
        """
        if self.k_cache is None:
            # 首次: 直接缓存
            self.k_cache = k
            self.v_cache = v
        else:
            # 后续: 拼接到已有缓存末尾（时间维 dim=2）
            self.k_cache = torch.cat([self.k_cache, k], dim=2)
            self.v_cache = torch.cat([self.v_cache, v], dim=2)
        return self.k_cache, self.v_cache

    def reset(self):
        """清空缓存（开始新的序列时调用）"""
        self.k_cache = None
        self.v_cache = None


# ===========================================================================
# Multi-Head Self-Attention — 多头自注意力（支持 GQA + KV Cache）
# ===========================================================================

class Attention(nn.Module):
    """多头自注意力层

    支持：
    - Grouped-Query Attention (GQA): K/V 头数可以少于 Q 头数
      例: 32 Q heads, 8 KV heads → 4x 复用，减少 75% KV Cache
    - KV Cache: 自回归生成时缓存历史 K/V
    - Flash Attention: 通过 PyTorch SDPA 加速

    计算流程:
    1. x → Wq, Wk, Wv 线性投影 → Q, K, V
    2. Q, K 应用 RoPE 旋转位置编码
    3. 如有 KV Cache，拼接历史 K/V
    4. GQA: 复制 K/V 头以匹配 Q 头数
    5. Attention(Q, K, V) = softmax(QK^T / √d) V
    6. 输出投影 Wo

    Args:
        config: 模型配置对象
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads                          # Q 头数
        self.head_dim = config.head_dim                        # 每头维度
        self.n_kv_heads = getattr(config, "n_kv_heads", config.n_heads)  # K/V 头数（GQA）
        self.n_rep = self.n_heads // self.n_kv_heads           # 每个 KV 头被几个 Q 头共享

        # 线性投影层（无偏置，现代 LLM 的标准做法）
        self.wq = nn.Linear(config.d_model, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, config.d_model, bias=False)

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.use_flash = config.use_flash_attention

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """将 K/V 头复制 n_rep 次以匹配 Q 头数（GQA 核心操作）

        例: n_kv_heads=8, n_heads=32, n_rep=4
        输入:  (batch, 8, seq_len, head_dim)
        输出:  (batch, 32, seq_len, head_dim)  每个 KV 头复制 4 份
        """
        if n_rep == 1:
            return x  # MHA（标准多头注意力），无需复制
        bs, n_kv_heads, seq_len, head_dim = x.shape
        return (
            x[:, :, None, :, :]                                # (bs, n_kv, 1, seq, d)
            .expand(bs, n_kv_heads, n_rep, seq_len, head_dim)  # (bs, n_kv, n_rep, seq, d)
            .reshape(bs, n_kv_heads * n_rep, seq_len, head_dim) # (bs, n_heads, seq, d)
        )

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        cache: Optional[KVCache] = None,
        layer_idx: int = 0,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向传播

        Args:
            x:         (batch, seq_len, d_model)        输入隐状态
            rope:      RotaryEmbedding                   旋转位置编码模块
            cache:     KVCache 或 None                   KV 缓存（生成时使用）
            layer_idx: int                               层索引（用于 KV Cache）
            mask:      (1, 1, seq_len, seq_len) 或 None  因果掩码

        Returns:
            (batch, seq_len, d_model) 注意力输出
        """
        B, T, _ = x.shape

        # ---- Step 1: 线性投影 → Q, K, V ----
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)      # (B, n_heads, T, d)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)   # (B, n_kv, T, d)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)   # (B, n_kv, T, d)

        # ---- Step 2: 应用 RoPE 旋转位置编码 ----
        cos, sin = rope(T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # ---- Step 3: KV Cache 更新 ----
        if cache is not None:
            k, v = cache.update(k, v, layer_idx)

        # ---- Step 4: GQA — 复制 K/V 头 ----
        k = self._repeat_kv(k, self.n_rep)
        v = self._repeat_kv(v, self.n_rep)

        # ---- Step 5: 计算注意力 ----
        if self.use_flash and hasattr(F, "scaled_dot_product_attention"):
            # PyTorch 2.0+ Flash Attention: 自动选择最优内核（Flash2 / Memory-Efficient）
            attn = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
        else:
            # 手动实现: QK^T / sqrt(d) → softmax → mask → @ V
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, n_heads, T, T_cached)
            if mask is not None:
                scores = scores + mask  # 因果掩码: 上三角为 -inf，softmax 后为 0
            weights = F.softmax(scores.float(), dim=-1).type_as(q)  # 转 float32 计算以保证精度
            weights = self.dropout(weights)
            attn = torch.matmul(weights, v)  # (B, n_heads, T, head_dim)

        # ---- Step 6: 合并多头 + 输出投影 ----
        attn = attn.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, n_heads * head_dim)
        return self.wo(attn)


# ===========================================================================
# SwiGLU Feed-Forward Network
# ===========================================================================

class FeedForward(nn.Module):
    """SwiGLU 前馈网络

    标准 FFN: output = W2(ReLU(W1(x)))
    SwiGLU:   output = W2(SiLU(W1(x)) * W3(x))

    SiLU (Swish) = x * sigmoid(x)，比 ReLU 更平滑
    门控机制: W3(x) 作为门控信号，与 SiLU(W1(x)) 逐元素相乘

    参数量: 3 × d_model × intermediate_size（比标准 FFN 多一个 W3）
    但效果更好，因此在相同参数预算下可以适当缩小 intermediate_size。

    LLaMA 的 intermediate_size 通常为 d_model 的 2.7~3.5 倍。
    例: d_model=4096, intermediate_size=11008 (约 2.7x)

    Args:
        config: 模型配置，需包含 d_model 和 intermediate_size
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.intermediate_size, bias=False)   # 上投影 + SiLU
        self.w3 = nn.Linear(config.d_model, config.intermediate_size, bias=False)   # 上投影 + 门控
        self.w2 = nn.Linear(config.intermediate_size, config.d_model, bias=False)   # 下投影
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        → w1(x): (batch, seq_len, intermediate_size) → SiLU 激活
        → w3(x): (batch, seq_len, intermediate_size) → 门控
        → 逐元素相乘 → w2 下投影回 d_model
        """
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ===========================================================================
# Decoder Block — 解码器块（Pre-Norm 结构）
# ===========================================================================

class DecoderBlock(nn.Module):
    """单个 Transformer 解码器块

    采用 Pre-Norm 结构（先归一化再计算子层）：
        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))

    比 Post-Norm（先计算子层再归一化）训练更稳定，梯度更平滑。
    所有现代 LLM（LLaMA, Qwen, Mistral, Gemma）都使用 Pre-Norm。

    Args:
        config: 模型配置
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)   # 注意力前的 RMSNorm
        self.attn = Attention(config)                                 # 多头自注意力
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)     # FFN 前的 RMSNorm
        self.ffn = FeedForward(config)                                # SwiGLU 前馈网络

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        cache: Optional[KVCache] = None,
        layer_idx: int = 0,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:         (batch, seq_len, d_model)
            rope:      旋转位置编码
            cache:     KV 缓存
            layer_idx: 层索引
            mask:      因果掩码

        Returns:
            (batch, seq_len, d_model) 经过注意力 + FFN + 残差后的输出
        """
        # 注意力子层: Norm → Attention → 残差
        x = x + self.attn(self.attn_norm(x), rope, cache, layer_idx, mask)
        # FFN 子层: Norm → FFN → 残差
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ===========================================================================
# Decoder-Only Transformer — 完整模型
# ===========================================================================

class DecoderOnlyTransformer(nn.Module):
    """完整的 Decoder-Only Transformer 模型

    整体数据流:
    token_ids → Embedding → Dropout → [DecoderBlock × N] → FinalNorm → LM Head → logits

    支持两种推理模式：
    1. 训练/Prefill: 一次性处理整个序列，输出所有位置的 logits
    2. 自回归生成: 逐 token 生成，使用 KV Cache 加速

    参数量计算（tie_embeddings=True 时）:
    = Embedding + N × (Attention + FFN) + FinalNorm
    = vocab × d + N × (4 × d² + 3 × d × intermediate) + d
    LLaMA-7B: 32000×4096 + 32×(4×4096² + 3×4096×11008) ≈ 6.7B

    Args:
        config: ModelConfig 实例
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token 嵌入层: token id → d_model 维向量
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # N 层 Decoder Block
        self.layers = nn.ModuleList([
            DecoderBlock(config) for _ in range(config.n_layers)
        ])

        # 最终 RMSNorm（输出前归一化）
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)

        # LM Head: d_model → vocab_size（语言模型头）
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # RoPE 旋转位置编码（全局共享，所有层使用相同的频率表）
        self.rope = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        # 权重共享: Embedding 和 LM Head 共享权重，减少参数量
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        # 参数初始化: 正态分布 N(0, 0.02)
        self.apply(self._init_weights)
        self._n_params = sum(p.numel() for p in self.parameters())

    def _init_weights(self, module: nn.Module):
        """权重初始化

        - Linear: 正态分布 N(0, 0.02)，偏置置零
        - Embedding: 正态分布 N(0, 0.02)
        0.02 是 LLaMA/GPT 系列的常用初始化标准差
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def n_params(self) -> int:
        """模型总参数量"""
        return self._n_params

    @staticmethod
    def _make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """创建因果掩码（下三角矩阵）

        确保位置 i 只能看到位置 ≤ i 的信息（自回归性质）

        例: seq_len=4
        [[  0, -inf, -inf, -inf],
         [  0,    0, -inf, -inf],
         [  0,    0,    0, -inf],
         [  0,    0,    0,    0]]

        -inf 在 softmax 后变为 0，即"看不到"
        """
        return torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_list: Optional[list[KVCache]] = None,
    ) -> torch.Tensor:
        """前向传播

        Args:
            input_ids:  (batch, seq_len) 输入 token id 序列
            cache_list: 每层一个 KVCache，用于自回归生成。None 表示训练模式。

        Returns:
            logits: (batch, seq_len, vocab_size) 每个位置的词表分布
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, \
            f"序列长度 {T} 超过最大限制 {self.config.max_seq_len}"

        # ---- Token Embedding ----
        x = self.dropout(self.token_embedding(input_ids))  # (B, T, d_model)

        # ---- 构建因果掩码 ----
        # Flash Attention 不需要显式掩码（SDPA 内部处理）
        # 手动注意力需要掩码来屏蔽未来位置
        mask = None
        if not self.config.use_flash_attention:
            if cache_list is not None and cache_list[0].k_cache is not None:
                # 生成模式: 已有缓存，掩码需覆盖 (新token, 全部缓存长度)
                cache_len = cache_list[0].k_cache.shape[2]
                mask = torch.triu(
                    torch.full((T, cache_len), float("-inf"), device=x.device),
                    diagonal=cache_len - T + 1  # 只能看到最近 T 个位置
                )
            else:
                # 训练/Prefill: 标准 (T, T) 因果掩码
                mask = self._make_causal_mask(T, x.device)
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T 或 T, cache_len)

        # ---- 通过 N 层 Decoder Block ----
        for i, layer in enumerate(self.layers):
            cache = cache_list[i] if cache_list is not None else None
            x = layer(x, self.rope, cache, layer_idx=i, mask=mask)

        # ---- 输出 ----
        x = self.final_norm(x)            # 最终归一化
        logits = self.lm_head(x)          # 投影到词表维度
        return logits                     # (B, T, vocab_size)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """自回归生成文本

        流程:
        1. Prefill: 一次性处理 prompt，初始化 KV Cache
        2. Decode: 逐 token 生成，每步只计算最新 token，利用 KV Cache

        采样策略:
        - temperature: 控制随机性。越高越随机，越低越确定。
          0.0 = 贪心（取概率最大的），1.0 = 正常，>1.0 更随机
        - top_k: 只从概率最高的 k 个 token 中采样
        - top_p (nucleus): 只从累计概率达到 p 的最小 token 集中采样
        - 两者可组合使用

        Args:
            input_ids:       (batch, seq_len) prompt 的 token id
            max_new_tokens:  最大生成 token 数
            temperature:     采样温度
            top_k:           Top-K 采样的 K 值，None 表示不使用
            top_p:           Nucleus 采样的累积概率阈值，1.0 表示不使用
            eos_token_id:    停止 token 的 id（遇到即停止生成）

        Returns:
            (batch, prompt_len + generated_len) 完整的 token id 序列
        """
        self.eval()
        # 每层创建一个 KV Cache
        cache_list = [KVCache() for _ in range(self.config.n_layers)]
        generated = input_ids

        # ---- Prefill: 处理整个 prompt ----
        logits = self.forward(input_ids, cache_list)
        next_logit = logits[:, -1, :]  # 只取最后一个位置的 logits

        for _ in range(max_new_tokens):
            # ---- 温度缩放 ----
            logits = next_logit / max(temperature, 1e-8)

            # ---- Top-K 过滤 ----
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # ---- Top-P (Nucleus) 过滤 ----
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # 移除累积概率超过 top_p 的 token（保留第一个超出的）
                remove_mask = cum_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[remove_mask] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            # ---- 采样 ----
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)
            generated = torch.cat([generated, next_token], dim=1)

            # ---- 检查 EOS ----
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            # ---- Decode: 只计算新 token，利用 KV Cache ----
            logits = self.forward(next_token, cache_list)
            next_logit = logits[:, -1, :]

        return generated
