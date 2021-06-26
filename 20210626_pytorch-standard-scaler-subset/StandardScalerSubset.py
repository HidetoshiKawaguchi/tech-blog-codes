# -*- coding: utf-8 -*-
'''
author: 川口 英俊 (Hidetoshi Kawaguchi)
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
'''
import torch
from torch.utils.data import Subset

class StandardScalerSubset(Subset):
    def __init__(self, dataset, indices,
                 mean=None, std=None, eps=10**-9):
        super().__init__(dataset=dataset, indices=indices)
        target_tensor = torch.stack([dataset[i][0] for i in indices])
        target_tensor = target_tensor.to(torch.float64)
        if mean is None:
            self._mean = torch.mean(target_tensor, dim=0)
        else:
            self._mean = mean
        if std is None:
            self._std = torch.std(target_tensor, dim=0, unbiased=False)
        else:
            self._std = std
        self._eps = eps
        self.std.apply_(lambda x: max(x, self.eps)) # ゼロ割対策

    def __getitem__(self, idx):
        dataset_list = list(self.dataset[self.indices[idx]])
        input = dataset_list[0]
        dataset_list[0] = (input - self.mean) / self.std
        return tuple(dataset_list)

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    @property
    def eps(self):
        return self._eps


if __name__ == '__main__':
    class MyDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.data = torch.tensor([[10, 100, 1000],
                                      [20,  50, 1500],
                                      [30, 150, 2500],
                                      [15, 175, 1300]])
            self.labels = tuple([1, 0, 1, 1])

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]


    dataset = MyDataset()
    # 0,1,2を訓練用データとする．
    train_sss = StandardScalerSubset(dataset, [0, 1, 2])
     # テストデータは，訓練時の平均と標準偏差で標準化を行う．
    test_sss = StandardScalerSubset(dataset, [3],
                                    mean=train_sss.mean, std=train_sss.std)
    print("Training data")
    for i in range(len(train_sss)):
        print(train_sss[i])
    print()
    print("Test data")
    print(test_sss[0])
