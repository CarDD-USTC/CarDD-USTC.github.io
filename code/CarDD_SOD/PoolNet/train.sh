#! /bin/bash

#python main.py --arch resnet --train_root ./data/DUTS/DUTS-TR --train_list ./data/DUTS/DUTS-TR/train_pair.lst

python main.py --arch resnet --mode train --device 1 --data_root /workspace/wangxinkuang/data/saliency_detection/CarDD --save_folder /workspace/wangxinkuang/model/PoolNet/CarDD
# you can optionly change the -lr and -wd params
