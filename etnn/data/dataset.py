from torch.utils.data import Dataset
from torch import Tensor
from etnn.data.permutation import TreeNode
from etnn.data.dataset_generation import generate_simple_multiclass_permutation
import numpy as np


class SimplePermutation(Dataset):
    def __init__(
            self,
            permutation_tree: TreeNode
    ):
        self.data, self.labels = generate_simple_multiclass_permutation(
            permutation_tree=permutation_tree,
            num_classes=2,
            integer_generation=True
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        x = self.data[item]
        if type(x) != np.ndarray:
            x = np.array([x])

        y = self.labels[item]
        if type(y) != np.ndarray:
            y = np.array([y])
        return Tensor(x), Tensor(y)
