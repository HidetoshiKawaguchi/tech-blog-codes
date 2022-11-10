# -*- coding: utf-8 -*-
'''
ImageNetのtrain用tarを展開後，中にある全てのtarをImageFolderで読み込める形で展開するためのスクリプト
'''
import os
import glob
import tarfile

if __name__ == '__main__':
    target_dir = './ILSVRC2012_img_train/'
    for tar_filepath in glob.glob(os.path.join(target_dir, '*.tar')):
        target_dir = tar_filepath.replace('.tar', '')
        os.mkdir(target_dir)
        with tarfile.open(tar_filepath, 'r') as tar:
            tar.extractall(path=target_dir)
        os.remove(tar_filepath) # 展開したtarファイルを削除
