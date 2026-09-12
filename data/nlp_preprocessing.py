import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        vocab = set()
        for s in positive + negative:
            vocab.update(s.split()) # .update() adds all elements from a list into the set
        
        word_to_id = {word: idx +1 for idx,word in enumerate(sorted(vocab))}

        encoded = [torch.tensor([word_to_id[w] for w in s.split()]) for s in positive + negative]

        return torch.nn.utils.rnn.pad_sequence(encoded,padding_value=0, batch_first=True)