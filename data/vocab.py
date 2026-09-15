from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        chars = set(list(text))
        chars = sorted(list(chars))
        stoi = {val: i for i, val in enumerate(chars)}
        itos = {i: val for i, val in enumerate(chars)}

        return (stoi, itos)

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        text = list(text)
        return [stoi[s] for s in text]

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        ids = list(ids)
        return ''.join(itos[i] for i in ids)
