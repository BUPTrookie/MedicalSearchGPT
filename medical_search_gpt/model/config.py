"""模型配置模块

定义 Decoder-Only Transformer 的全部超参数。
默认参数与 LLaMA-7B 对齐，可根据需要调整为 Qwen2、Mistral 等架构。
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Transformer Decoder-Only 模型配置

    Attributes:
        vocab_size:       词表大小。BPE 训练后的词表条目数，特殊 token + 256 字节 + 合并数。
                          LLaMA-1 为 32000，Qwen2 为 151936。
        max_seq_len:      最大序列长度，决定 RoPE 缓存大小和位置编码覆盖范围。
                          LLaMA-2 为 4096，Qwen2.5 支持 32768 + YaRN 扩展。
        d_model:          隐藏维度（模型宽度），所有子层的输入输出维度。
                          0.5B ≈ 1024, 7B ≈ 4096, 70B ≈ 8192。
        n_heads:          查询(Query)头数。注意力头数 * 头维度 = d_model。
        n_layers:         Decoder Block 堆叠层数。模型深度，越多容量越大但越慢。
                          0.5B ≈ 24, 7B ≈ 32, 70B ≈ 80。
        intermediate_size: FFN 中间层维度。SwiGLU 的投影目标维度。
                          通常为 d_model * 2.7 ~ 3.5（如 11008 = 4096 * ~2.7）。
        rope_theta:       RoPE 旋转位置编码的底数 θ。越大则远程衰减越慢，
                          有利于长上下文外推。LLaMA 默认 10000，Qwen2.5 用 1000000。
        rope_ratio:       RoPE 频率缩放因子。>1 时相当于位置压缩，用于上下文扩展。
                          NTK-aware scaling / YaRN 等方法中会用到。
        norm_eps:         RMSNorm 中的 ε，防止除零。LLaMA 用 1e-6，Qwen 用 1e-5。
        dropout:          Dropout 率。预训练通常 0.0（不用），微调时可设 0.1。
        tie_embeddings:   是否共享 token_embedding 和 lm_head 的权重。
                          True = 共享（减少参数量），False = 各自独立。
        use_flash_attention: 是否使用 PyTorch 2.0+ 的 SDPA flash attention。
                          True = 更快更省显存，False = 手动实现注意力（便于调试）。
    """

    vocab_size: int = 32000
    max_seq_len: int = 4096
    d_model: int = 4096
    n_heads: int = 32
    n_layers: int = 32
    intermediate_size: int = 11008
    rope_theta: float = 10000.0
    rope_ratio: float = 1.0
    norm_eps: float = 1e-6
    dropout: float = 0.0
    tie_embeddings: bool = True
    use_flash_attention: bool = False

    @property
    def head_dim(self) -> int:
        """每个注意力头的维度 = d_model / n_heads"""
        return self.d_model // self.n_heads
