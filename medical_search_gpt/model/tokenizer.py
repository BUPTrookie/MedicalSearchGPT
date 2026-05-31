"""Byte Pair Encoding (BPE) 分词器

完整的 BPE 分词器实现，支持：
1. 从文本训练词表（train）
2. 编码文本为 token id 序列（encode）
3. 解码 token id 序列为文本（decode）
4. 持久化词表到文件（save / load）

核心流程：
    原始文本 → GPT-4 正则预分词 → UTF-8 字节映射 → BPE 合并 → token id 序列

字节映射策略（与 GPT-2/4 相同）：
    将 0~255 的字节值映射到可打印 Unicode 字符，避免控制字符干扰 BPE 合并。
    可打印 ASCII 和 Latin-1 字符保持不变，其余字节映射到 256+ 的 Unicode 码位。
"""

import regex
from pathlib import Path
import json
from typing import Optional


# GPT-4 的预分词正则表达式
# 按以下优先级将文本切分为块（chunks）：
#   1. 英文缩写：'s, 't, 're, 've, 'm, 'll, 'd（不区分大小写）
#   2. 字母序列（可带前导非字母数字字符）：如 " hello", "world"
#   3. 数字序列（最多 3 位一组）：如 "123", "45"
#   4. 标点/符号序列：如 "!!!", "---"
#   5. 换行符（带前导空白）
#   6. 尾部空白
# 目的：让 BPE 在语义合理的边界上合并，避免跨词合并
GPT4_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


def _get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    """获取一个 token 序列中所有相邻的二元组。

    例: ("h", "e", "l", "l", "o") → {("h","e"), ("e","l"), ("l","l"), ("l","o")}
    用于在 BPE 训练中统计可合并的 token 对。
    """
    return {(word[i], word[i + 1]) for i in range(len(word) - 1)}


class BPETokenizer:
    """BPE 分词器

    完整的生命周期：
        1. tokenizer = BPETokenizer()
        2. tokenizer.train(corpus_text, vocab_size=32000)  # 训练
        3. ids = tokenizer.encode("你好世界")                 # 编码
        4. text = tokenizer.decode(ids)                     # 解码
        5. tokenizer.save("tokenizer.json")                 # 保存

    特殊 token：
        <pad> (0) — 批量填充，训练时忽略
        <bos> (1) — 序列起始标记
        <eos> (2) — 序列结束标记，生成时遇到即停止
        <unk> (3) — 未知 token 回退
    """

    SPECIAL_TOKENS = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}

    def __init__(self):
        self.merges: dict[tuple[str, str], int] = {}          # 合并规则: (token_a, token_b) → 优先级
        self.vocab: dict[str, int] = {}                       # token 字符串 → id
        self.inverse_vocab: dict[int, str] = {}               # id → token 字符串
        self._pat = regex.compile(GPT4_PATTERN)               # 预编译正则
        self._byte_encoder = self._build_byte_encoder()       # 字节 → Unicode 映射
        self._byte_decoder = {v: k for k, v in self._byte_encoder.items()}  # 反向映射
        self._initialized = False                              # 是否已完成训练/加载

    @staticmethod
    def _build_byte_encoder() -> dict[int, str]:
        """构建字节到 Unicode 字符的映射表。

        将 256 个字节值映射为可打印字符：
        - ASCII 可打印字符 (! ~) 和 Latin-1 扩展字符 (¡ ¬, ® ÿ) 直接映射到自身
        - 其余字节（控制字符等）映射到 256+ 的 Unicode 码位

        这样 BPE 处理的所有 token 都是可见字符，便于调试和序列化。
        """
        # 可打印字节范围
        bs = list(range(ord("!"), ord("~") + 1)) + \
             list(range(ord("¡"), ord("¬") + 1)) + \
             list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        # 不可打印字节映射到 256+
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return {b: chr(c) for b, c in zip(bs, cs)}

    def _text_to_tokens(self, text: str) -> list[str]:
        """使用 GPT-4 正则将文本预切分为块。

        例: "Hello world! 123" → ["Hello", " world", "!", " 123"]
        每个 chunk 将独立进行字节编码和BPE处理。
        """
        return list(self._pat.findall(text))

    def train(self, text: str, vocab_size: int = 32000, verbose: bool = False):
        """从文本语料训练 BPE 词表

        算法流程：
        1. 预分词 → 字节编码 → 统计每个 word 的频率
        2. 迭代合并：每次找频率最高的相邻 token 对，合并为新 token
        3. 重复直到达到目标词表大小

        Args:
            text: 训练语料文本
            vocab_size: 目标词表大小（含特殊 token + 256 字节 + BPE 合并数）
            verbose: 是否打印训练进度
        """
        num_merges = vocab_size - 256 - len(self.SPECIAL_TOKENS)
        assert num_merges > 0, f"vocab_size 太小，至少需要 {256 + len(self.SPECIAL_TOKENS)}"

        # ---- Step 1: 预处理 ----
        # 将文本预分词后，每个 chunk 转为字节序列再映射为 Unicode token
        # 统计每个唯一 word（tuple of tokens）的出现频率
        chunks = self._text_to_tokens(text)
        word_freqs: dict[tuple[str, ...], int] = {}
        for chunk in chunks:
            encoded = tuple(self._byte_encoder[b] for b in chunk.encode("utf-8"))
            word_freqs[encoded] = word_freqs.get(encoded, 0) + 1

        # ---- Step 2: 迭代合并 ----
        merges: dict[tuple[str, str], int] = {}
        for i in range(num_merges):
            # 统计所有 word 中相邻 token 对的加权频率
            pair_counts: dict[tuple[str, str], int] = {}
            for word, freq in word_freqs.items():
                pairs = _get_pairs(word)
                for p in pairs:
                    pair_counts[p] = pair_counts.get(p, 0) + freq
            if not pair_counts:
                break  # 所有 token 已合并完毕

            # 选择频率最高的 token 对
            best = max(pair_counts, key=pair_counts.get)
            merges[best] = i

            # 在所有 word 中执行合并：(a, b) → ab
            merged = best[0] + best[1]
            new_freqs: dict[tuple[str, ...], int] = {}
            for word, freq in word_freqs.items():
                new_word: list[str] = []
                j = 0
                while j < len(word):
                    if j < len(word) - 1 and word[j] == best[0] and word[j + 1] == best[1]:
                        new_word.append(merged)
                        j += 2
                    else:
                        new_word.append(word[j])
                        j += 1
                new_freqs[tuple(new_word)] = new_freqs.get(tuple(new_word), 0) + freq
            word_freqs = new_freqs

            if verbose and (i + 1) % 1000 == 0:
                print(f"merge {i + 1}/{num_merges}: {best} -> {merged}")

        self.merges = merges

        # ---- Step 3: 构建词表 ----
        # 词表 = 特殊 token + 256 个基础字节 + 所有 BPE 合并结果
        self.vocab = dict(self.SPECIAL_TOKENS)
        for i in range(256):
            self.vocab[self._byte_encoder[i]] = len(self.vocab)
        for (a, b), idx in sorted(merges.items(), key=lambda x: x[1]):
            self.vocab[a + b] = len(self.vocab)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self._initialized = True

    def _bpe(self, token: str) -> list[str]:
        """对单个预编码 token 执行 BPE 合并

        按训练好的合并规则优先级，反复将最高优先级的相邻 token 对合并，
        直到没有可合并的 token 对为止。

        例: "Ġhello" → ["Ġh", "e", "l", "lo"]（取决于训练结果）
        """
        word = tuple(token)
        if len(word) <= 1:
            return list(word)
        while True:
            pairs = _get_pairs(word)
            if not pairs:
                break
            # 选择合并优先级最高的 token 对（优先级最低的索引 = 最先学到的合并）
            best = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if best not in self.merges:
                break  # 没有可合并的 token 对
            # 执行合并
            merged = best[0] + best[1]
            new_word: list[str] = []
            j = 0
            while j < len(word):
                if j < len(word) - 1 and word[j] == best[0] and word[j + 1] == best[1]:
                    new_word.append(merged)
                    j += 2
                else:
                    new_word.append(word[j])
                    j += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
        return list(word)

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> list[int]:
        """将文本编码为 token id 序列

        流程：
        1. GPT-4 正则预分词 → 切分为 chunks
        2. 每个 chunk → UTF-8 字节 → Unicode 映射
        3. 对 Unicode 字符串执行 BPE 合并
        4. 查词表转为 id

        Args:
            text: 待编码文本
            add_bos: 是否在开头添加 <bos> token
            add_eos: 是否在结尾添加 <eos> token

        Returns:
            token id 列表，如 [1, 3847, 2988, ...]（1 = <bos>）
        """
        assert self._initialized, "分词器未训练，请先调用 train() 或 load()"
        ids: list[int] = []
        if add_bos:
            ids.append(self.SPECIAL_TOKENS["<bos>"])

        for chunk in self._text_to_tokens(text):
            # chunk → UTF-8 字节 → Unicode 映射后的字符串
            byte_encoded = "".join(self._byte_encoder[b] for b in chunk.encode("utf-8"))
            # 对映射后的字符串执行 BPE 合并
            tokens = self._bpe(byte_encoded)
            for t in tokens:
                ids.append(self.vocab.get(t, self.SPECIAL_TOKENS["<unk>"]))

        if add_eos:
            ids.append(self.SPECIAL_TOKENS["<eos>"])
        return ids

    def decode(self, ids: list[int]) -> str:
        """将 token id 序列解码为文本

        流程：
        1. 跳过 <bos>、<pad>，遇到 <eos> 停止
        2. id → Unicode token 字符串
        3. 拼接所有 token → 逐字符反向映射为字节 → UTF-8 解码

        Args:
            ids: token id 列表

        Returns:
            解码后的文本字符串
        """
        assert self._initialized, "分词器未训练，请先调用 train() 或 load()"
        tokens = []
        for i in ids:
            if i in (self.SPECIAL_TOKENS["<bos>"], self.SPECIAL_TOKENS["<pad>"]):
                continue
            if i == self.SPECIAL_TOKENS["<eos>"]:
                break
            tokens.append(self.inverse_vocab.get(i, ""))
        # 拼接 token → Unicode 字符串 → 字节 → UTF-8 文本
        text = "".join(tokens)
        bs = bytearray([self._byte_decoder[c] for c in text])
        return bs.decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        """当前词表大小"""
        return len(self.vocab)

    def save(self, path: str | Path):
        """将合并规则保存到 JSON 文件

        只保存 merges（合并规则），词表可以在 load 时从 merges 重建。
        """
        data = {
            "merges": {f"{k[0]}||{k[1]}": v for k, v in self.merges.items()},
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def load(self, path: str | Path):
        """从 JSON 文件加载合并规则并重建词表"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.merges = {tuple(k.split("||")): v for k, v in data["merges"].items()}
        # 从 merges 重建词表
        self.vocab = dict(self.SPECIAL_TOKENS)
        for i in range(256):
            self.vocab[self._byte_encoder[i]] = len(self.vocab)
        for (a, b), idx in sorted(self.merges.items(), key=lambda x: x[1]):
            self.vocab[a + b] = len(self.vocab)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self._initialized = True
