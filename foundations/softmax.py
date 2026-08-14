import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        z_less_max = z - np.max(z)
        e_power = np.exp(z_less_max)
        total = np.sum(e_power)
        return np.round(e_power / total, 4)