import tiktoken as _tiktoken

from tokenizers.base import Tokenizer


class TiktokenTokenizer(Tokenizer):
    """Tokenizer backed by tiktoken (e.g. cl100k_base, o200k_base)."""

    def __init__(self, encoding: str = "cl100k_base"):
        self._enc = _tiktoken.get_encoding(encoding)

    def fit(self, text: str) -> None:
        pass 

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab
