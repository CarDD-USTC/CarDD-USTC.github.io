import os
from PIL import Image
import cv2
import torch
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional as F
import numbers
import numpy as np
import random


class ImageDataTrain(data.Dataset):
    def __init__(self, data_root, data_list):
        self.sal_root = data_root
        self.sal_source = data_list
        
        self.img_list = os.listdir(os.path.join(self.sal_root, 'CarDD-TR-Image'))
        self.gt_list = os.listdir(os.path.join(self.sal_root, 'CarDD-TR-Mask'))

        self.img_list.sort()
        self.gt_list.sort()
        print(self.img_list[0:10])
        print(self.gt_list[0:10])
        self.sal_num = len(self.img_list)


    def __getitem__(self, item):
        # sal data loading
        im_name = self.img_list[item % self.sal_num]
        gt_name = self.gt_list[item % self.sal_num]
        
        sal_image = load_image(os.path.join(self.sal_root,'CarDD-TR-Image' ,im_name))
        sal_label = load_sal_label(os.path.join(self.sal_root,'CarDD-TR-Mask' ,gt_name))
        # sal_edge = load_sal_label(os.path.join(self.sal_root,'CarDD-TR-Edge' ,gt_name.replace('.png','_edge.png')))
        sal_edge = load_sal_label(os.path.join(self.sal_root,'CarDD-TR-Edge' ,gt_name))

        sal_image, sal_label,sal_edge = cv_random_flip(sal_image, sal_label,sal_edge)
        sal_image = sal_image.transpose((1, 2, 0))
        sal_label = sal_label.transpose((1, 2, 0))
        sal_edge = sal_edge.transpose((1, 2, 0))
        sal_image, sal_label, sal_edge = generate_scale_label(sal_image, sal_label, sal_edge)
        sal_image, sal_label, sal_edge = random_rotate(sal_image, sal_label, sal_edge)
        sal_image = sal_image.transpose((2, 0, 1))
        sal_label = sal_label.transpose((2, 0, 1))
        sal_edge = sal_edge.transpose((2, 0, 1))


        sal_saliency = np.squeeze(sal_label)
        kernelll = np.ones((25,25),np.uint8)
        sal_saliency = cv2.dilate(sal_saliency, kernelll, 2)
        kernel_size = (25, 25)
        sal_saliency = cv2.GaussianBlur(sal_saliency, kernel_size, 8)
        sal_saliency = sal_saliency/np.max(sal_saliency)
        sal_saliency = np.expand_dims(sal_saliency, axis=0)

        sal_image = torch.Tensor(sal_image)
        sal_label = torch.Tensor(sal_label)
        sal_edge = torch.Tensor(sal_edge)
        sal_saliency = torch.Tensor(sal_saliency)
        

        sample = {'sal_image': sal_image, 'sal_label': sal_label, 'sal_edge': sal_edge, 'sal_saliency': sal_saliency}
        return sample

    def __len__(self):
        return self.sal_num


class ImageDataTest(data.Dataset):
    def __init__(self, data_root, data_list):
        self.data_root = data_root
        self.data_list = data_list
        with open(self.data_list, 'r') as f:
            self.image_list = [x.strip() for x in f.readlines()]

        self.image_num = len(self.image_list)

    def __getitem__(self, item):
        image, im_size = load_image_test(os.path.join(self.data_root, self.image_list[item]))
        image = torch.Tensor(image)

        return {'image': image, 'name': self.image_list[item % self.image_num], 'size': im_size}

    def __len__(self):
        return self.image_num


def get_loader(config, mode='train', pin=False):
    shuffle = False
    if mode == 'train':
        shuffle = True
        dataset = ImageDataTrain(config.train_root, config.train_list)
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_thread, pin_memory=pin)
    else:
        dataset = ImageDataTest(config.test_root, config.test_list)
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_thread, pin_memory=pin)
    return data_loader

def load_image(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path)
    in_ = np.array(im, dtype=np.float32)
    in_ -= np.array((104.00699, 116.66877, 122.67892))
    in_ = in_.transpose((2,0,1))
    return in_

def load_image_test(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path)
    in_ = np.array(im, dtype=np.float32)
    im_size = tuple(in_.shape[:2])
    in_ -= np.array((104.00699, 116.66877, 122.67892))
    in_ = in_.transpose((2,0,1))
    return in_, im_size

def load_sal_label(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = Image.open(path)
    label = np.array(im, dtype=np.float32)
    if len(label.shape) == 3:
        label = label[:,:,0]
    label = label / 255.
    label = label[np.newaxis, ...]
    return label

def cv_random_flip(img, label, edge):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img[:,:,::-1].copy()
        label = label[:,:,::-1].copy()
        edge = edge[:,:,::-1].copy()
    return img, label,edge

def generate_scale_label(image, label, edge):
    flip_flag = random.randint(0, 1)

    if flip_flag == 1:
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        edge = cv2.resize(edge, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        h, w, c = image.shape
        #image = np.reshape(image, (h, w, 3))
        label = np.reshape(label, (h, w, 1))
        edge = np.reshape(edge, (h, w, 1))
    return image, label, edge

def random_rotate(x,y,z):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        angle = np.random.randint(-25,25)
        #print('x',x.shape)
        h, w, c = x.shape
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        x = cv2.warpAffine(x, M, (w, h))
        y = cv2.warpAffine(y, M, (w, h))
        z = cv2.warpAffine(z, M, (w, h))
        y = np.reshape(y, (h, w, 1))
        z = np.reshape(z, (h, w, 1))
    return x, y, z

def random_crop(x,y,z):
    h,w = y.shape
    randh = np.random.randint(h/8)
    randw = np.random.randint(w/8)
    #randf = np.random.randint(10)
    offseth = 0 if randh == 0 else np.random.randint(randh)
    offsetw = 0 if randw == 0 else np.random.randint(randw)
    p0, p1, p2, p3 = offseth,h+offseth-randh, offsetw, w+offsetw-randw
    return x[p0:p1,p2:p3],y[p0:p1,p2:p3],z[p0:p1,p2:p3]