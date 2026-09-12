import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        # Layers: Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid

        # can't use Sequential since averaging step breaks the chain
        self.emb = nn.Embedding(vocabulary_size, 16)
        self.linear = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer

        # Return a B, 1 tensor and round to 4 decimal places
        ret = self.emb(x)
        # T is the sequence length — the number of tokens in each input sentence.
        #the full shape (B, T, 16) means:
        #B  = number of sentences in the batch
        #T  = number of tokens in each sentence (after padding)
        #16 = embedding dimension — one 16-dim vector per token
        ret = torch.mean(ret,dim=1)
        # input of linear is the innput size
        ret = self.linear(ret)
        ret = self.sigmoid(ret)

        return ret.round(decimals=4)
