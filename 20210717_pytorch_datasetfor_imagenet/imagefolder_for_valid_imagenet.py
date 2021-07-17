# -*- coding: utf-8 -*-
from torchvision import transforms
from torchvision.datasets import ImageFolder

if __name__ == '__main__':
    # ImageNetの訓練データのパス(valid_root), ImageFolderのrootに設定
    valid_root = './ILSVRC2012_img_val_for_ImageFolder'

    # ImageFolderの前処理, ImageFolderのtransoformに設定
    valid_transform = transforms.Compose([
        transforms.Resize(224), # 1辺が224ピクセルの正方形に変換
        transforms.ToTensor()   # Tensor行列に変換
    ])

    # ImageFolderのインスタンス生成
    validset = ImageFolder(root=valid_root, # 画像が保存されているフォルダのパス
                           transform=valid_transform) # Tensorへの変換

    # 動作確認
    img, label = validset[1]
    print('img = ', img)
    print('class(WordNet ID) = ', validset.classes[label])
