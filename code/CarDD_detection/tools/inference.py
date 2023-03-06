# Author: wxk
# Time: 2021/11/26 15:02

import sys
from mmdet.apis import init_detector, inference_detector
import os
import shutil
import argparse
import time
import warnings

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet inference')
    parser.add_argument(
        '--img-path',
        help='the source path')
    parser.add_argument(
        '--save-path',
        help='the destination path')
    parser.add_argument(
        '--config-file',
        help='the config file path')
    parser.add_argument(
        '--checkpoint-file',
        help='the checkpoint file path')
    parser.add_argument(
        '--cuda',
        choices=['0', '1'],
        default='0',
        help='which cuda')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    img_path = args.img_path
    save_path = args.save_path
    config_file = args.config_file
    checkpoint_file = args.checkpoint_file
    model = init_detector(config_file, checkpoint_file, device='cuda:' + args.cuda)
    PALETTE = [(255, 182, 193), (0, 168, 225), (0, 255, 0), (128, 0, 128), (255, 255, 0), (227, 0, 57)]  # change color

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('working directory is', save_path.split('/')[-1])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    source_imgs = sorted(os.listdir(img_path))
    dest_imgs = sorted(os.listdir(save_path))
    cnt = len(dest_imgs) + 1
    for img in source_imgs:
        print('predicting', cnt, 'image:', img)
        if not img.endswith('.jpg') or img in dest_imgs:
            cnt += 1
            continue
        result = inference_detector(model, os.path.join(img_path, img))
        model.show_result(os.path.join(img_path, img), result, out_file=os.path.join(save_path, img), score_thr=0.5, text_color=PALETTE, font_size=20)
        cnt += 1


if __name__ == '__main__':
    main()
