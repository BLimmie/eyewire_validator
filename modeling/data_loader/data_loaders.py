# from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
import json
import os
import copy

from tools.create_data import create_indiv_perm_stack, create_stack

# class MnistDataLoader(BaseDataLoader):
#     """
#     MNIST data loading demo using BaseDataLoader
#     """
#     def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
#         trsfm = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#             ])
#         self.data_dir = data_dir
#         self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
#         super(MnistDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        
class EyewireDataset(Dataset):
    def __init__(self, data_path, prev=0):
        """
        args

        data_path: path/to/data

        data_path contains
        - task_vol.json
        - task_data.json
        - images/
        - segmentation/ 
        """
        with open(os.path.join(data_path, 'task_vol.json')) as f:
            j = json.load(f)
        self.data = sorted(copy.copy(list(j.keys())))
        self.data_path = data_path
        self.prev = prev
        j = None

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return create_stack(self.data[idx], self.data_path, self.prev)

class EyewireDataLoader(BaseDataLoader):
    def __init__(self, data_path, batch_size, shuffle, validation_split, num_workers, prev=0):
        self.data_path = data_path
        self.dataset = EyewireDataset(self.data_path, prev)
        super(EyewireDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)