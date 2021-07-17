# -*- coding: utf-8 -*-
from torchvision import transforms
from torchvision.datasets import ImageFolder

if __name__ == '__main__':
    # ImageNetの訓練データのパス(train_root), ImageFolderのrootに設定
    train_root = './ILSVRC2012_img_train'

    # ImageFolderの前処理, ImageFolderのtransoformに設定
    train_transform = transforms.Compose([
        transforms.Resize(224), # 1辺が224ピクセルの正方形に変換
        transforms.ToTensor()   # Tensor行列に変換
    ])

    # ImageFolderのインスタンス生成
    trainset = ImageFolder(root=train_root, # 画像が保存されているフォルダのパス
                           transform=train_transform) # Tensorへの変換

    # 動作確認
    img, label = trainset[1]
    print('img = ', img)
    print('class(WordNet ID) = ', trainset.classes[label])
