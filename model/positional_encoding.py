import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        # PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        #
        # Hint: Use np.arange() to create position and dimension index vectors,
        # then compute all values at once with broadcasting (no loops needed).
        # Assign sine to even columns (PE[:, 0::2]) and cosine to odd columns (PE[:, 1::2]).

        pos = np.arange(seq_len)   # shape (seq_len,)  → [0, 1, 2, ..., seq_len-1]
        i   = np.arange(d_model // 2)  # shape (d_model//2,) → [0, 1, 2, ..., d_model//2 - 1]

        pos = pos.reshape(seq_len, 1)      # shape (seq_len, 1) that tells broadcating this are the rows
        angles = pos / (10000 ** (2 * i / d_model))  # shape (seq_len, d_model//2)
        PE = np.zeros((seq_len,d_model))
        PE[:, 0::2] = np.round(np.sin(angles),5)   # even dims
        PE[:, 1::2] = np.round(np.cos(angles),5)   # odd dims

        return PE


        # Round to 5 decimal places.
        
