import os
import torch
from torchvision.datasets import SVHN as PyTorchSVHN
import numpy as np


class SVHN:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=8):

        # to fit with repo conventions for location
        modified_location = os.path.join(location, 'svhn')

        self.train_dataset = PyTorchSVHN(
            root=modified_location,
            download=True,
            split='train',
            transform=preprocess
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        self.test_dataset = PyTorchSVHN(
            root=modified_location,
            download=True,
            split='test',
            transform=preprocess
        )
        SPLIT = True
        if SPLIT:
            test_subset_ids = np.random.choice(len(self.test_dataset), 2000, replace=False).flatten()
            self.test_dataset  = torch.utils.data.Subset(
                self.test_dataset,
                indices=test_subset_ids
            )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        self.classnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
