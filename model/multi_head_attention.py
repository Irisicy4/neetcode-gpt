import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.k_proj = nn.Linear(embedding_dim,attention_dim,bias=False)
        self.q_proj = nn.Linear(embedding_dim,attention_dim,bias=False)
        self.v_proj = nn.Linear(embedding_dim,attention_dim,bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places

        k = self.k_proj(embedded)
        q = self.q_proj(embedded)
        v = self.v_proj(embedded)

        #attention score
        ret = q @ k.transpose(-2, -1)   # also .T only works on 2D, use transpose for batched
        # divide by attention_dim, which is q.shape[-1] (last dim)
        ret = ret / (q.shape[-1] ** 0.5)

        mask = torch.tril(torch.ones(embedded.shape[1], embedded.shape[1]))
        ret = ret.masked_fill(mask == 0, float('-inf'))
        # Why float('-inf') and not 0 — after softmax, 0 would still contribute to attention. float('-inf') becomes exactly 0 after softmax (e^-inf = 0), meaning future tokens are completely ignored.
        
        #dim=2 (or equivalently dim=-1) applies softmax along the last dimension — across the columns of each row, which is exactly what you want.
        ret = ret.softmax(dim=2)

        return (ret @ v).round(decimals=4)



        


        
