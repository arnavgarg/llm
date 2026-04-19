from abc import ABC, abstractmethod


class Tokenizer(ABC):
    """Abstract base class for tokenizers."""

    @abstractmethod
    def fit(self, text: str) -> None:
        """Build vocabulary from text."""

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Convert text to a list of token IDs."""

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        """Convert a list of token IDs back to text."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary."""
