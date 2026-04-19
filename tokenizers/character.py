from tokenizers.base import Tokenizer


class CharacterTokenizer(Tokenizer):
    """Tokenizer that maps each unique character to an integer ID."""

    def __init__(self):
        self.char_to_id: dict[str, int] = {}
        self.id_to_char: dict[int, str] = {}

    def fit(self, text: str) -> None:
        """Build vocabulary from text."""
        for char in sorted(set(text)):
            if char not in self.char_to_id:
                idx = len(self.char_to_id)
                self.char_to_id[char] = idx
                self.id_to_char[idx] = char

    def encode(self, text: str) -> list[int]:
        """Convert text to a list of token IDs."""
        return [self.char_to_id[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        """Convert a list of token IDs back to text."""
        return "".join(self.id_to_char[i] for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.char_to_id)
