# Author: wxk
# Time: 2022/2/11 17:55


"""
	需要修改的地方:
		dataDir,savepath改为自己的路径
		class_names改为自己需要的类
		dataset_list改为自己的数据集名称
"""
from pycocotools.coco import COCO
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import math
from shutil import copyfile


# 生成保存路径，函数抄的(›´ω`‹ )
# if the dir is not exists,make it,else delete it
def mkr(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)


# 生成mask图
def mask_generator(coco, width, height, anns_list):
    mask_pic = np.zeros((height, width))
    # 生成mask - 此处生成的是4通道的mask图,如果使用要改成三通道,可以将下面的注释解除,或者在使用图片时搜相关的程序改为三通道
    for single in anns_list:
        mask_single = coco.annToMask(single)
        mask_pic += mask_single
    # 转化为255
    for row in range(height):
        for col in range(width):
            if (mask_pic[row][col] > 0):
                mask_pic[row][col] = 255
    mask_pic = mask_pic.astype(int)
    # return mask_pic

    # 转为三通道
    imgs = np.zeros(shape=(height, width, 3), dtype=np.float32)
    imgs[:, :, 0] = mask_pic[:, :]
    imgs[:, :, 1] = mask_pic[:, :]
    imgs[:, :, 2] = mask_pic[:, :]
    imgs = imgs.astype(np.uint8)
    return imgs


def edge_generator(coco, width, height, anns_list):
    edge_pic = np.zeros((height, width))
    for single in anns_list:
        seg = single['segmentation'][0]
        points = []
        for i in range(int(len(seg) / 2)):
            points.append([seg[2 * i], seg[2 * i + 1]])
        for i in range(len(points)):
            if i == len(points) - 1:
                j = 0
            else:
                j = i + 1
            if points[j][0] > points[i][0]:
                xs = range(round(points[i][0]), min(round(points[j][0]) + 1, width - 1), 1)
                k = (points[j][1] - points[i][1]) / (points[j][0] - points[i][0])
            elif points[j][0] < points[i][0]:
                xs = range(min(round(points[i][0]), width - 1), max(round(points[j][0]) - 1, -1), -1)
                k = (points[j][1] - points[i][1]) / (points[j][0] - points[i][0])
            else:
                xs = []
                k = 0
            for x in xs:
                y = points[i][1] + k * (x - points[i][0])
                # if y >= height or x >= width:
                #     print('zzz')
                try:
                    edge_pic[min(math.floor(y), height - 1), x] = 255
                    edge_pic[min(math.ceil(y), height - 1), x] = 255
                except:
                    print(111)
                # if math.floor(y) > 0 and math.floor(y) < height:
                #     try:
                #         edge_pic[x,math.floor(y)] = 255
                #     except:
                #         print(111)
                # if math.ceil(y) > 0 and math.ceil(y) < height:
                #     try:
                #         edge_pic[x,math.ceil(y)] = 255
                #     except:
                #         print(111)
        for i in range(len(points)):
            if i == len(points) - 1:
                j = 0
            else:
                j = i + 1
            if points[j][1] > points[i][1]:
                ys = range(round(points[i][1]), min(round(points[j][1]) + 1, height - 1), 1)
                k = (points[j][0] - points[i][0]) / (points[j][1] - points[i][1])
            elif points[j][1] < points[i][1]:
                ys = range(min(round(points[i][1]), height - 1), max(round(points[j][1]) - 1, -1), -1)
                k = (points[j][0] - points[i][0]) / (points[j][1] - points[i][1])
            else:
                ys = []
                k = 0
            for y in ys:
                x = points[i][0] + k * (y - points[i][1])
                # if y>=height or x >=width:
                #     print('hhh')
                try:
                    edge_pic[y, min(math.floor(x), width - 1)] = 255
                    edge_pic[y, min(math.ceil(x), width - 1)] = 255
                except:
                    print(222)
                # if math.floor(x) > 0 and math.floor(x) < height:
                #     edge_pic[math.floor(x),y] = 255
                # if math.ceil(x) > 0 and math.ceil(x) < height:
                #     edge_pic[math.ceil(x),y] = 255
        # print(max(seg))
        # print(width)
        # print(height)
        # print(111)

    imgs = np.zeros(shape=(height, width, 3), dtype=np.float32)
    imgs[:, :, 0] = edge_pic[:, :]
    imgs[:, :, 1] = edge_pic[:, :]
    imgs[:, :, 2] = edge_pic[:, :]
    imgs = imgs.astype(np.uint8)
    return imgs


# 处理json数据并保存二值mask
def get_mask_data(annFile, mask_to_save, edge_to_save, classes_names, mask_mode):
    # 获取COCO_json的数据
    coco = COCO(annFile)
    # 拿到所有需要的图片数据的id - 我需要的类别的categories的id是多少
    classes_ids = coco.getCatIds(catNms=classes_names)
    # 取所有类别的并集的所有图片id
    # 如果想要交集，不需要循环，直接把所有类别作为参数输入，即可得到所有类别都包含的图片

    if mask_mode == 'subset':
        imgIds_list = []
        # 循环取出每个类别id对应的有哪些图片并获取图片的id号
        for idx in classes_ids:
            imgidx = coco.getImgIds(catIds=idx)  # 将该类别的所有图片id好放入到一个列表中
            imgIds_list += imgidx
            print("搜索id... ", imgidx)
        # 去除重复的图片
        imgIds_list = list(set(imgIds_list))  # 把多种类别对应的相同图片id合并
    else:
        imgIds_list = coco.getImgIds()

    # 一次性获取所有图像的信息
    image_info_list = coco.loadImgs(imgIds_list)

    # 对每张图片生成一个mask
    for imageinfo in image_info_list:
        # 获取对应类别的分割信息
        annIds = coco.getAnnIds(imgIds=imageinfo['id'], catIds=classes_ids, iscrowd=None)
        anns_list = coco.loadAnns(annIds)
        # 生成二值mask图
        mask_image = mask_generator(coco, imageinfo['width'], imageinfo['height'], anns_list)
        edge_image = edge_generator(coco, imageinfo['width'], imageinfo['height'], anns_list)
        mask_name = mask_to_save + '/' + imageinfo['file_name'][:-4] + '.png'
        edge_name = edge_to_save + '/' + imageinfo['file_name'][:-4] + '.png'
        plt.imsave(mask_name, mask_image)
        plt.imsave(edge_name, edge_image)
        print("已保存Mask: ", mask_name, "已保存edge: ", edge_name)


def generate_lst_file(root_path, TR_or_TE):
    file_list = sorted(os.listdir(os.path.join(root_path, TR_or_TE, TR_or_TE + '-Image')))
    if TR_or_TE == 'CarDD-TR':
        f = open(os.path.join(root_path, TR_or_TE) + '/train_pair.lst', "a")
        for file in file_list:
            f.write('CarDD-TR-Image/' + file + 'CarDD-TR-Mask/' + file.replace('jpg', 'png'))
            f.write('\n')
        f.close()
    elif TR_or_TE == 'CarDD-TE' or 'CarDD-VAL':
        f = open(os.path.join(root_path, TR_or_TE) + '/test.lst', "a")
        for file in file_list:
            f.write(file)
            f.write('\n')
        f.close()
    else:
        print('input error!')


def generate_dataset(savepath, classes_names, mask_mode):
    for dataset in datasets_list:
        mask_to_save = os.path.join(savepath, datasets_dict[dataset], datasets_dict[dataset] + '-Mask')
        edge_to_save = os.path.join(savepath, datasets_dict[dataset], datasets_dict[dataset] + '-Edge')
        image_to_save = os.path.join(savepath, datasets_dict[dataset], datasets_dict[dataset] + '-Image')
        mkr(mask_to_save)
        mkr(edge_to_save)
        mkr(image_to_save)
        annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataset)
        get_mask_data(annFile, mask_to_save, edge_to_save, classes_names, mask_mode)
        print('Got all the masks of {} from {}'.format(classes_names, dataset))

        for mask in os.listdir(mask_to_save):
            img = mask.replace('.png', '.jpg')
            copyfile(os.path.join(dataDir, dataset, img), os.path.join(image_to_save, img))

        generate_lst_file(savepath, datasets_dict[dataset])


if __name__ == '__main__':
    '''
    路径参数
    '''
    dataset_mode = 'classwise'  # classwise:每类单独生成一个文件夹 其他:包含所有类别
    # dataset_mode = 'all'  # classwise:每类单独生成一个文件夹 其他:包含所有类别
    # mask_mode = 'subset'  # subset:只保存含损伤的图片  #其他:生成所有图片的mask
    mask_mode = 'all'  # subset:只保存含损伤的图片  #其他:生成所有图片的mask
    dataDir = r"/workspace/wangxinkuang/data/Shutterstock_coco_v7"
    datasets_dict = {'train2017': 'CarDD-TR', 'val2017': 'CarDD-VAL', 'test2017': 'CarDD-TE'}
    datasets_list = ['train2017', 'val2017', 'test2017']
    # datasets_list = ['test2017']

    if dataset_mode == 'classwise':
        classes_list = ['dent', 'scratch', 'crack', 'glass shatter', 'lamp broken', 'tire flat']
        for cls in classes_list:
            savepath = os.path.join("/workspace/wangxinkuang/data/CarDD-classwise", 'CarDD-' + cls.replace(' ', '-'))
            classes_names = [cls]
            generate_dataset(savepath, classes_names, mask_mode)

    else:
        savepath = r"/workspace/wangxinkuang/data/CarDD/"
        classes_names = ['dent', 'scratch', 'crack', 'glass shatter', 'lamp broken', 'tire flat']
        generate_dataset(savepath, classes_names, mask_mode)
