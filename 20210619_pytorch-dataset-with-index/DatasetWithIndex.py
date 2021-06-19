# -*- coding: utf-8 -*-
'''
author: 川口 英俊 (Hidetoshi Kawaguchi)
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
'''

class DatasetWithIndex:
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, label = self.dataset[index]
        return data, label, index

    def __len__(self):
        return len(self.dataset)

    @property
    def classes(self):
        return self.dataset.classes


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision import transforms as tt
    from torchvision.datasets import MNIST

    dataset = MNIST(root='./', train=True, download=True,
                    transform=tt.Compose([tt.ToTensor()]))
    dataset_with_index = DatasetWithIndex(dataset) # ★データセットをラップしている
    data_loader = DataLoader(dataset_with_index, batch_size=4, shuffle=True)

    # デモンストレーション1
    ## 一部データを取得し，あとで取得したインデックスで同じデータにアクセスできるか調べる．
    print("デモンストレーション1")
    input_list, label_list, index_list = [], [], []
    for i, data in enumerate(data_loader):
        inputs, labels, indices = data
        input_list.extend(inputs)
        label_list.extend(labels)
        index_list.extend(indices)
        if i >= 3:
            break
    for input, label, index in zip(input_list, label_list, index_list):
        data = dataset_with_index[index]
        # indexの辻褄があっているかを確認
        assert (input == data[0]).all()
        assert data[1] == label
        print("label1 = {}, label2= {}".format(data[1], label))
    print("len(dataset_with_index) = {}".format(len(dataset_with_index)))
    print("dataset_with_index.classes = {}".format(dataset_with_index.classes))


    # デモンストレーション2: Subset
    print()
    print(print("デモンストレーション2: Subsetとの組み合わせ"))
    from torch.utils.data import Subset

    ## Subset上のインデックスを取得する
    ## SubsetをDatasetWithIndexでラップする
    subset1 = Subset(dataset, indices=[2, 1, 3, 5, 4])
    subset_with_index = DatasetWithIndex(subset1)
    print('index on a subset = {}'.format(subset_with_index[0][2]))

    ## 元のデータセットのインデックスを取得する
    ## DatasetWithIndexをSubsetでラップする
    subset_with_raw_index = Subset(dataset_with_index, [2, 1, 3, 5, 4])
    print('index on a raw dataset = {}'.format(subset_with_raw_index[0][2]))
