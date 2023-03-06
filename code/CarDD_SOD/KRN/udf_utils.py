# Author: wxk
# Time: 2022/1/17 16:00
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
import shutil


def visualization(img, save_path=''):
    for i in range(img.shape[1]):
        show_img = img[0, i, :, :]
        print(show_img.shape)
        show_img = show_img.cpu()
        array1 = show_img.detach().numpy()
        # maxValue = array1.max()
        # array1 = array1 * 255 / maxValue
        mat = np.uint8(array1)
        # mat = mat.transpose(1, 2, 0)
        # mat = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)
        # cv2.imshow('img', mat)
        cv2.imwrite(save_path.replace('.jpg', '_' + str(i) + '.jpg'), mat)
        cv2.waitKey(0)


def mkr(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)


def mask_overlap(image_path, mask_path, anno_path, save_path):
    mkr(save_path)

    mask_colors = ['None', 'yellow', 'red']
    anno_colors = ['white', 'yellow', 'deepskyblue']
    bounds = [0, 128, 256]

    mask_cmap = mpl.colors.ListedColormap(mask_colors)
    anno_cmap = mpl.colors.ListedColormap(anno_colors)
    norm = mpl.colors.BoundaryNorm(bounds, mask_cmap.N)

    img_list = os.listdir(image_path)
    img_list.sort()
    for img in img_list:
        print('drawing', img, '...')
        image = Image.open(os.path.join(image_path, img))
        mask = Image.open(os.path.join(mask_path, img.replace('jpg', 'png')))
        anno = Image.open(os.path.join(anno_path, img.replace('jpg', 'png')))
        anno = np.asarray(anno)[:, :, 0]  # annotation使用

        plt.figure(figsize=(7.2, 4.8))
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)

        plt.imshow(image, aspect='auto')
        plt.imshow(anno, aspect='auto', alpha=0.5, interpolation='none', cmap=anno_cmap, norm=norm)  # alpha - 用于控制透明的程度
        plt.imshow(mask, aspect='auto', alpha=0.5, interpolation='none', cmap=mask_cmap, norm=norm)  # alpha - 用于控制透明的程度
        plt.savefig(os.path.join(save_path, img.replace('jpg', 'png')))
        plt.close()


import os
from sklearn.metrics import mean_absolute_error, precision_recall_curve, precision_recall_fscore_support
import numpy as np
import cv2
import matplotlib.pyplot as plt
from copy import deepcopy


def evalutate(mask_file, anno_file):
    mask = cv2.imread(mask_file)[:, :, 0]
    mask = mask.astype(int)
    anno = cv2.imread(anno_file)[:, :, 0]
    anno = anno.astype(int)
    size = mask.shape
    mask = mask.reshape(size[0] * size[1], )
    anno = anno.reshape(size[0] * size[1], )
    mae_mode = 'threshold'  # threshold:设置阈值缩放到0和1，precise:计算像素差mae/255
    # mae_mode = 'precise'  # threshold:设置阈值缩放到0和1，precise:计算像素差mae/255

    precision, recall, thresholds = precision_recall_curve(anno, mask, pos_label=255)
    Fscore = []
    for i in range(len(precision)):
        if precision[i] == 0 or recall[i] == 0:
            continue
        Fscore.append(2 / ((1 / precision[i]) + (1 / recall[i])))
    Fmax = max(Fscore)
    Favg = np.mean(Fscore)
    print('Fmax:', Fmax)
    print('Favg:', Favg)

    MAE = mean_absolute_error(anno, mask) / 255
    print('MAE:', MAE)

    return Fmax, Favg, MAE


def evalutate_saliency(mask_path, anno_path):
    anno_files = os.listdir(anno_path)
    anno_files.sort()
    Fmax_list = []
    Favg_list = []
    MAE_list = []
    count = 0

    for file in anno_files:
        print('\nevaluating', file)
        mask_file = os.path.join(mask_path, file)
        anno_file = os.path.join(anno_path, file)
        Fmax, Favg, MAE = evalutate(mask_file, anno_file)
        Fmax_list.append(Fmax)
        Favg_list.append(Favg)
        MAE_list.append(MAE)
        count += 1
    print(count, 'files evaluated')
    print('\nTOTAL Fmax:', np.mean(Fmax_list), 'TOTAL Favg:', np.mean(Favg_list), 'TOTAL MAE:', np.mean(MAE_list))


def sum_saliency_map():
    path_all = '/workspace/wangxinkuang/model/saliency_detection/sgl_krn/CarDD-01/pred'
    path_save = '/workspace/wangxinkuang/model/saliency_detection/sgl_krn/CarDD-01/pred-sum'
    path_dent = '/workspace/wangxinkuang/model/saliency_detection/sgl_krn/CarDD-dent/pred-CarDD-dent'
    path_scratch = '/workspace/wangxinkuang/model/saliency_detection/sgl_krn/CarDD-scratch/pred-CarDD-scratch'
    path_crack = '/workspace/wangxinkuang/model/saliency_detection/sgl_krn/CarDD-crack/pred-CarDD-crack'
    path_glass_shatter = '/workspace/wangxinkuang/model/saliency_detection/sgl_krn/CarDD-glass-shatter/pred-CarDD-glass-shatter'
    path_lamp_broken = '/workspace/wangxinkuang/model/saliency_detection/sgl_krn/CarDD-lamp-broken/pred-CarDD-lamp-broken'
    path_tire_flat = '/workspace/wangxinkuang/model/saliency_detection/sgl_krn/CarDD-tire-flat/pred-CarDD-tire-flat'
    mkr(path_save)
    list_path = [path_dent, path_scratch, path_crack, path_glass_shatter, path_lamp_broken, path_tire_flat]
    for file in sorted(os.listdir(path_all)):
        print('summing',file)
        img = cv2.imread(os.path.join(path_all, file))[:, :, 0]
        (height, width) = img.shape
        slice = np.zeros((height, width))
        for path in list_path:
            if file in os.listdir(path):
                pred = cv2.imread(os.path.join(path, file))[:, :, 0]
                slice = np.maximum(slice, pred)  # 比较两个array，逐个元素取最大
        sum_img = np.zeros(shape=(height, width, 3), dtype=np.float32)
        sum_img[:, :, 0] = slice[:, :]
        sum_img[:, :, 1] = slice[:, :]
        sum_img[:, :, 2] = slice[:, :]
        sum_img = sum_img.astype(np.uint8)
        plt.imsave(os.path.join(path_save, file), sum_img)

if __name__ == '__main__':
    # image_path = '/workspace/wangxinkuang/data/saliency_detection/CarDD-glass-shatter/CarDD-TE/CarDD-TE-Image'
    # mask_path = '/workspace/wangxinkuang/model/saliency_detection/sgl_krn/CarDD-glass-shatter/pred-CarDD-glass-shatter'
    # anno_path = '/workspace/wangxinkuang/data/saliency_detection/CarDD-glass-shatter/CarDD-TE/CarDD-TE-Mask'
    # save_path = '/workspace/wangxinkuang/data/saliency_detection/CarDD-glass-shatter/visualization'
    # mask_overlap(image_path, mask_path, anno_path, save_path)

    # mask_path = '/workspace/wangxinkuang/result/2022-02-23-01/od2sod_tire_flat'
    # anno_path = '/workspace/wangxinkuang/data/saliency_detection/CarDD-tire-flat/CarDD-TE/CarDD-TE-Mask'
    # # # mask_path = '/workspace/wangxinkuang/model/saliency_detection/sgl_krn/CarDD-01/pred'
    # # # anno_path = '/workspace/wangxinkuang/data/saliency_detection/CarDD/CarDD-TE/CarDD-TE-Mask'

    # mask_path = '/workspace/wangxinkuang/result/2022-02-26-01/U2Net_CarDD'
    # anno_path = '/workspace/wangxinkuang/data/saliency_detection/CarDD/CarDD-TE/CarDD-TE-Mask'

    mask_path = '/workspace/wangxinkuang/result/U2Net'
    anno_path = '/workspace/wangxinkuang/data/saliency_detection/CarDD/CarDD-TE/CarDD-TE-Mask'
    evalutate_saliency(mask_path, anno_path)

    # sum_saliency_map()
