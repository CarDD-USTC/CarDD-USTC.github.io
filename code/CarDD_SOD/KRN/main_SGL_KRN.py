import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import math
from torch.autograd import Variable
import numpy as np
from udf_utils import visualization

from networks.deeplab_resnet import \
    resnet50_locate
from networks.vgg import vgg16_locate

config_vgg = {'convert': [[128, 256, 512, 512, 512], [64, 128, 256, 512, 512]],
              'deep_pool': [[512, 512, 256, 128], [512, 256, 128, 128], [True, True, True, False],
                            [True, True, True, False]], 'score': 128}  # no convert layer, no conv6

config_resnet = {'convert': [[64, 256, 512, 1024, 2048], [128, 256, 256, 512, 512]],
                 'deep_pool': [[512, 512, 256, 256, 128], [512, 256, 256, 128, 128], [False, True, True, True, False],
                               [True, True, True, True, False]], 'score': 128}

save_path = r'/home/wangxinkuang/Files/research/code/KRN/results/'

class ConvertLayer(nn.Module):
    def __init__(self, list_k):
        super(ConvertLayer, self).__init__()
        up = []
        for i in range(len(list_k[0])):
            up.append(nn.Sequential(nn.Conv2d(list_k[0][i], list_k[1][i], 1, 1, bias=False), nn.ReLU(inplace=True)))
        self.convert0 = nn.ModuleList(up)

    def forward(self, list_x):
        resl = []
        for i in range(len(list_x)):
            resl.append(self.convert0[i](list_x[i]))
        return resl


class DeepPoolLayer_first(nn.Module):
    def __init__(self, k, k_out, need_x2,
                 need_fuse):  # (config['deep_pool'][0][i], config['deep_pool'][1][i], config['deep_pool'][2][i], config['deep_pool'][3][i])
        super(DeepPoolLayer_first, self).__init__()
        self.pools_sizes = [2, 2, 2]
        self.need_x2 = need_x2
        self.need_fuse = need_fuse
        pools, convs = [], []
        for i in self.pools_sizes:
            pools.append(nn.AvgPool2d(kernel_size=i, stride=i))
            convs.append(nn.Conv2d(k, k, 3, 1, 1, bias=False))
        self.pools = nn.ModuleList(pools)
        self.convs = nn.ModuleList(convs)
        self.relu = nn.ReLU()
        self.conv_sum = nn.Conv2d(k, k_out, 3, 1, 1, bias=False)
        if self.need_fuse:
            self.conv_sum_c = nn.Conv2d(k_out, k_out, 3, 1, 1, bias=False)

    def forward(self, x, x2=None):  # (merge, conv2merge[k+1], infos[k])
        x_size = x.size()
        resl = x
        # visualization(resl, save_path + 'resl.jpg')
        y = x
        for i in range(len(self.pools_sizes)):
            try:
                y = self.convs[i](self.pools[i](y))
            except:
                return ValueError
            resl = torch.add(resl, F.interpolate(y, x_size[2:], mode='bilinear', align_corners=True))
        resl = self.relu(resl)
        if self.need_x2:  # 这个是上菜样的过程
            resl = F.interpolate(resl, x2.size()[2:], mode='bilinear', align_corners=True)
        resl = self.conv_sum(resl)
        if self.need_fuse:
            resl = self.conv_sum_c(torch.add(resl, x2))
        return resl


class ScoreLayer(nn.Module):
    def __init__(self, k):
        super(ScoreLayer, self).__init__()
        self.score = nn.Conv2d(k, 1, 1, 1)

    def forward(self, x, x_size=None):
        x = self.score(x)
        if x_size is not None:
            x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=True)
        return x


def extra_layer(base_model_cfg, vgg):
    if base_model_cfg == 'vgg':
        config = config_vgg
    elif base_model_cfg == 'resnet':
        config = config_resnet
    convert_layers, score_layers = [], []
    convert_layers = ConvertLayer(config['convert'])
    score_layers = ScoreLayer(config['score'])

    return vgg, convert_layers, score_layers


class KRN_edge(nn.Module):
    def __init__(self, base_model_cfg, base, convert_layers, score_layers):
        super(KRN_edge, self).__init__()
        self.base_model_cfg = base_model_cfg
        self.base = base

        #self.deep_pool = nn.ModuleList(deep_pool_layers)
        self.score = score_layers
        if self.base_model_cfg == 'resnet':
            self.convert = convert_layers

        # 'deep_pool': [[512, 512, 256, 256, 128], [512, 256, 256, 128, 128], [False, True, True, True, False], [True, True, True, True, False]]
        self.DeepPool_solid1 = DeepPoolLayer_first(512, 512, False, True)
        self.DeepPool_solid2 = DeepPoolLayer_first(512, 256, True, True)
        self.DeepPool_solid3 = DeepPoolLayer_first(256, 256, True, True)
        self.DeepPool_solid4 = DeepPoolLayer_first(256, 128, True, True)
        self.DeepPool_solid5 = DeepPoolLayer_first(128, 128, False, False)

        self.DeepPool_contour1 = DeepPoolLayer_first(512, 512, False, True)
        self.DeepPool_contour2 = DeepPoolLayer_first(512, 256, True, True)
        self.DeepPool_contour3 = DeepPoolLayer_first(256, 256, True, True)
        self.DeepPool_contour4 = DeepPoolLayer_first(256, 128, True, True)
        self.DeepPool_contour5 = DeepPoolLayer_first(128, 128, False, False)

        self.relu = nn.ReLU()
        self.conv_reduce1 = nn.Conv2d(512, 128, 1, 1, 1, bias=False)
        self.conv_reduce2 = nn.Conv2d(256, 128, 1, 1, 1, bias=False)
        self.conv_reduce3 = nn.Conv2d(256, 128, 1, 1, 1, bias=False)

        self.score_solid = ScoreLayer(128)
        self.score_contour = ScoreLayer(128)

        self.score_solid1 = ScoreLayer(512)
        self.score_contour1 = ScoreLayer(512)

        self.score_solid2 = ScoreLayer(256)
        self.score_contour2 = ScoreLayer(256)

        self.score_solid3 = ScoreLayer(256)
        self.score_contour3 = ScoreLayer(256)

        self.score_solid4 = ScoreLayer(128)
        self.score_contour4 = ScoreLayer(128)

        self.score_solid = ScoreLayer(128)
        self.score_contour = ScoreLayer(128)
        self.score_sum_out = ScoreLayer(128)


        self.conv_1 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.conv_2 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.conv_3 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.conv_4 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)

        self.conv_add1 = nn.Conv2d(128, 32, 3, 1, 1, bias=False)
        self.conv_add2 = nn.Conv2d(128, 32, 3, 1, 1, bias=False)
        self.conv_add3 = nn.Conv2d(128, 32, 3, 1, 1, bias=False)
        self.conv_add4 = nn.Conv2d(128, 32, 3, 1, 1, bias=False)
        self.conv_sum_out = nn.Conv2d(128, 128, 3, 1, 1, bias=False)

    def forward(self, x):
        x_size = x.size()
        conv2merge, infos = self.base(x)  #
        if self.base_model_cfg == 'resnet':
            conv2merge = self.convert(conv2merge)  # 将维度改变[64,256,512,1024,2048],[128,256,256,512,512]
        conv2merge = conv2merge[::-1]

        merge_contour1 = self.conv_1(conv2merge[1])
        merge_solid1 = self.DeepPool_solid1(conv2merge[0], conv2merge[1])
        if merge_solid1 == ValueError:
            return merge_solid1
        out_merge_solid1 = self.score_solid1(merge_solid1, x_size)
        out_merge_contour1 = self.score_contour1(merge_contour1, x_size)
        out_merge_solid1 = F.sigmoid(out_merge_solid1)
        out_merge_contour1 = F.sigmoid(out_merge_contour1)
        #merge_contour1, merge_solid1 = self.fuse1(merge_contour1, merge_solid1)#
        fea_reduce1 = self.conv_reduce1(merge_solid1)
        fea_reduce1 = self.relu(fea_reduce1)



        merge_contour2 = self.conv_2(conv2merge[2])
        merge_solid2 = self.DeepPool_solid2(merge_solid1, conv2merge[2])
        out_merge_solid2 = self.score_solid2(merge_solid2, x_size)
        out_merge_contour2 = self.score_contour2(merge_contour2, x_size)
        out_merge_solid2 = F.sigmoid(out_merge_solid2)
        out_merge_contour2 = F.sigmoid(out_merge_contour2)
        #merge_contour2, merge_solid2 = self.fuse2(merge_contour2, merge_solid2)  #
        fea_reduce2 = self.conv_reduce2(merge_solid2)
        fea_reduce2 = self.relu(fea_reduce2)



        merge_contour3 = self.conv_3(conv2merge[3])
        merge_solid3 = self.DeepPool_solid3(merge_solid2, conv2merge[3])
        out_merge_solid3 = self.score_solid3(merge_solid3, x_size)
        out_merge_contour3 = self.score_contour3(merge_contour3, x_size)
        out_merge_solid3 = F.sigmoid(out_merge_solid3)
        out_merge_contour3 = F.sigmoid(out_merge_contour3)
        #merge_contour3, merge_solid3 = self.fuse3(merge_contour3, merge_solid3)  #
        fea_reduce3 = self.conv_reduce3(merge_solid3)
        fea_reduce3 = self.relu(fea_reduce3)



        merge_contour4 = self.conv_4(conv2merge[4])
        merge_solid4 = self.DeepPool_solid4(merge_solid3, conv2merge[4])
        out_merge_solid4 = self.score_solid4(merge_solid4, x_size)
        out_merge_contour4 = self.score_contour4(merge_contour4, x_size)
        out_merge_solid4 = F.sigmoid(out_merge_solid4)
        out_merge_contour4 = F.sigmoid(out_merge_contour4)
        #merge_contour4, merge_solid4 = self.fuse4(merge_contour4, merge_solid4)  #
        fea_reduce4 = merge_solid4


        merge_solid5 = self.DeepPool_solid5(merge_solid4)
        merge_solid = self.score_solid(merge_solid5, x_size)  #
        merge_solid = F.sigmoid(merge_solid)

        fea_reduce1 = F.interpolate(fea_reduce1, merge_solid5.size()[2:], mode='bilinear', align_corners=True)
        fea_reduce2 = F.interpolate(fea_reduce2, merge_solid5.size()[2:], mode='bilinear', align_corners=True)
        fea_reduce3 = F.interpolate(fea_reduce3, merge_solid5.size()[2:], mode='bilinear', align_corners=True)
        fea_reduce4 = F.interpolate(fea_reduce4, merge_solid5.size()[2:], mode='bilinear', align_corners=True)
        fea_add1 = torch.add(merge_solid5, fea_reduce1)
        fea_add2 = torch.add(merge_solid5, fea_reduce2)
        fea_add3 = torch.add(merge_solid5, fea_reduce3)
        fea_add4 = torch.add(merge_solid5, fea_reduce4)
        fea_add1 = self.conv_add1(fea_add1)
        fea_add1 = self.relu(fea_add1)
        fea_add2 = self.conv_add2(fea_add2)
        fea_add2 = self.relu(fea_add2)
        fea_add3 = self.conv_add3(fea_add3)
        fea_add3 = self.relu(fea_add3)
        fea_add4 = self.conv_add4(fea_add4)
        fea_add4 = self.relu(fea_add4)
        feasum_out = torch.cat((fea_add1, fea_add2, fea_add3, fea_add4), 1)
        feasum_out = self.conv_sum_out(feasum_out)
        feasum_out = self.score_sum_out(feasum_out, x_size)  #
        feasum_out = F.sigmoid(feasum_out)


        return feasum_out, merge_solid,  out_merge_solid1, out_merge_contour1, out_merge_solid2, out_merge_contour2, out_merge_solid3, out_merge_contour3, out_merge_solid4, out_merge_contour4


def build_model(base_model_cfg='vgg'):
    if base_model_cfg == 'vgg':
        return KRN_edge(base_model_cfg, *extra_layer(base_model_cfg, vgg16_locate()))
    elif base_model_cfg == 'resnet':
        return KRN_edge(base_model_cfg, *extra_layer(base_model_cfg, resnet50_locate()))


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()




import torch
from collections import OrderedDict
from torch.nn import utils, functional as F
from torch.optim import Adam
from torch.autograd import Variable
from torch.backends import cudnn
import scipy.misc as sm
import numpy as np
import os
import torchvision.utils as vutils
import cv2
import math
import time


def bce2d(input, target, reduction=None):
    if not input.size() == target.size():
        print(input.shape)
        print(target.shape)
    assert (input.size() == target.size())
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg / num_total
    beta = 1.1 * num_pos / num_total
    # target pixel = 1 -> weight beta
    # target pixel = 0 -> weight 1-beta
    weights = alpha * pos + beta * neg

    return F.binary_cross_entropy(input, target, weights, reduction=reduction)

def _iou(pred, target, size_average = True):

    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        Iand1 = torch.sum(target[i,:,:,:]*pred[i,:,:,:])
        Ior1 = torch.sum(target[i,:,:,:]) + torch.sum(pred[i,:,:,:])-Iand1
        IoU1 = Iand1/Ior1

        #IoU loss is (1-IoU1)
        IoU = IoU + (1-IoU1)

    return IoU/b

class IOU(torch.nn.Module):
    def __init__(self, size_average = True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):

        return _iou(pred, target, self.size_average)
iou_loss = IOU(size_average=True)
class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.iter_size = config.iter_size
        self.show_every = config.show_every
        self.lr_decay_epoch = [15, ]
        self.build_model()
        if config.mode == 'test':
            if self.config.cuda:
                print('Loading pre-trained model from %s...' % self.config.test_model)
                self.net.load_state_dict(torch.load(self.config.test_model, map_location='cpu'))
            else:
                print('Loading pre-trained model from %s...' % self.config.model)
                self.net.load_state_dict(torch.load(self.config.model, map_location='cpu'))
            self.net.eval()

    # print the network information and parameter numbers
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    # build the network
    def build_model(self):
        self.net = build_model(self.config.arch)
        if self.config.cuda:
            self.net = self.net.cuda()
        # self.net.train()
        self.net.eval()  # use_global_stats = True
        self.net.apply(weights_init)
        if self.config.load == '':
            self.net.base.load_pretrained_model(torch.load(self.config.pretrained_model))
        else:
            self.net.load_state_dict(torch.load(self.config.load))

        self.lr = self.config.lr
        self.wd = self.config.wd

        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr,
                              weight_decay=self.wd)
        # self.print_network(self.net, 'KRN_edge Structure')

    def test(self):
        mode_name = 'sal_fuse'
        time_s = time.time()
        img_num = len(self.test_loader)
        for i, data_batch in enumerate(self.test_loader):
            images, name, im_size = data_batch['image'], data_batch['name'][0], np.asarray(data_batch['size'])
            print('predicting', name)
            with torch.no_grad():
                images = Variable(images)
                if self.config.cuda:
                    images = images.cuda()
                feasum_out, merge_solid,  out_merge_solid1, out_merge_contour1, out_merge_solid2, out_merge_contour2, out_merge_solid3, out_merge_contour3, out_merge_solid4, out_merge_contour4 = self.net(
                    images)

                pred = np.squeeze(feasum_out).cpu().data.numpy()
                multi_fuse = 255 * pred
                # cv2.imwrite(os.path.join(self.config.test_fold, name[:-4] + '_' + mode_name + '.png'), multi_fuse)
                cv2.imwrite(os.path.join(self.config.test_fold, name.replace('jpg', 'png')), multi_fuse)
        time_e = time.time()
        print('Speed: %f FPS' % (img_num / (time_e - time_s)))
        print('Test Done!')

    # training phase
    def train(self):
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        aveGrad = 0
        x_showEvery = 0
        for epoch in range(self.config.epoch):
            r_sal_loss = 0
            r_sal_loss1 = 0
            self.net.zero_grad()
            for i, data_batch in enumerate(self.train_loader):
                sal_image, sal_label, sal_edge, im_name = data_batch['sal_image'], data_batch['sal_label'], data_batch[
                    'sal_edge'], data_batch['im_name']
                # visualization(sal_image, save_path + 'sal_image.jpg')
                # visualization(sal_label, save_path + 'sal_label.jpg')
                # visualization(sal_edge, save_path + 'sal_edge.jpg')
                # print('sal_image0',sal_image.shape)
                # print('sal_label0', sal_label.shape)
                # print('sal_edge0', sal_edge.shape)
                if (sal_image.size(2) != sal_label.size(2)) or (sal_image.size(3) != sal_label.size(3)):
                    print('IMAGE ERROR, PASSING```')
                    continue
                sal_image, sal_label, sal_edge = Variable(sal_image), Variable(sal_label), Variable(sal_edge)
                # visualization(sal_image, save_path + 'sal_image_v.jpg')
                # visualization(sal_label, save_path + 'sal_label_v.jpg')
                # visualization(sal_edge, save_path + 'sal_edge_v.jpg')
                if self.config.cuda:
                    # cudnn.benchmark = True
                    sal_image, sal_label, sal_edge = sal_image.cuda(), sal_label.cuda(), sal_edge.cuda()

                try:
                    feasum_out, merge_solid,  out_merge_solid1, out_merge_contour1, out_merge_solid2, out_merge_contour2, out_merge_solid3, out_merge_contour3, out_merge_solid4, out_merge_contour4 = self.net(
                        sal_image)
                except (TypeError,UnboundLocalError) as e:
                    print(e,im_name)
                # visualization(feasum_out, save_path + 'feasum_out.jpg')
                # visualization(merge_solid, save_path + 'merge_solid.jpg')
                # visualization(out_merge_solid1, save_path + 'out_merge_solid1.jpg')

                feasum_out_loss = F.binary_cross_entropy(feasum_out, sal_label, reduction='mean') + iou_loss(feasum_out,
                                                                                                         sal_label)
                solid_loss = F.binary_cross_entropy(merge_solid, sal_label, reduction='mean') + iou_loss(merge_solid, sal_label)
                solid_loss1 = F.binary_cross_entropy(out_merge_solid1, sal_label, reduction='mean') + iou_loss(out_merge_solid1, sal_label)
                edge_loss1 = bce2d(out_merge_contour1, sal_edge, reduction='mean')
                solid_loss2 = F.binary_cross_entropy(out_merge_solid2, sal_label, reduction='mean') + iou_loss(out_merge_solid2, sal_label)
                edge_loss2 = bce2d(out_merge_contour2, sal_edge, reduction='mean')
                solid_loss3 = F.binary_cross_entropy(out_merge_solid3, sal_label, reduction='mean') + iou_loss(out_merge_solid3, sal_label)
                edge_loss3 = bce2d(out_merge_contour3, sal_edge, reduction='mean')
                solid_loss4 = F.binary_cross_entropy(out_merge_solid4, sal_label, reduction='mean') + iou_loss(out_merge_solid4, sal_label)
                edge_loss4 = bce2d(out_merge_contour4, sal_edge, reduction='mean')

                sal_loss = (
                                       2*feasum_out_loss + solid_loss + edge_loss1 + solid_loss1 + edge_loss2 + solid_loss2 + edge_loss3 + solid_loss3 + edge_loss4 + solid_loss4) / (
                                       self.iter_size * self.config.batch_size)
                r_sal_loss += solid_loss.data
                solid_loss1 = F.binary_cross_entropy(merge_solid, sal_label, reduction='sum')

                r_sal_loss1 += solid_loss1.data
                x_showEvery += 1

                sal_loss.backward()

                aveGrad += 1

                # accumulate gradients as done in DSS
                if aveGrad % self.iter_size == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    aveGrad = 0

                if i % (self.show_every // self.config.batch_size) == 0:
                    #                     if i == 0:
                    #                         x_showEvery = 1
                    print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  Sal : %10.4f ||  Sal1 : %10.4f' % (
                        epoch, self.config.epoch, i, iter_num, r_sal_loss / x_showEvery, r_sal_loss1 / x_showEvery))
                    # print('Learning rate: ' + str(self.lr))
                    r_sal_loss = 0
                    r_sal_loss1 = 0
                    x_showEvery = 0

            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(), '%s/models/epoch_%d.pth' % (self.config.save_folder, epoch + 1))

            if epoch in self.lr_decay_epoch:
                self.lr = self.lr * 0.1
                self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr,
                                      weight_decay=self.wd)

        torch.save(self.net.state_dict(), '%s/models/final.pth' % self.config.save_folder)



import argparse
import os
from dataset.dataset_edge_augment import get_loader

def get_test_info(sal_mode='e',data_root='/workspace/wangxinkuang/data/saliency_detection/CarDD'):
    if sal_mode == 'e':
        image_root = data_root + '/data/ECSSD/Imgs/'
        image_source = data_root + './data/ECSSD/test.lst'
    elif sal_mode == 'p':
        image_root = data_root + '/data/PASCALS/Imgs/'
        image_source = data_root + '/data/PASCALS/test.lst'
    elif sal_mode == 'd':
        image_root = data_root + '/data/DUTOMRON/Imgs/'
        image_source = data_root + '/data/DUTOMRON/test.lst'
    elif sal_mode == 'h':
        image_root = data_root + '/data/HKU-IS/Imgs/'
        image_source = data_root + '/data/HKU-IS/test.lst'
    elif sal_mode == 's':
        image_root = data_root + '/data/SOD/Imgs/'
        image_source = data_root + '/data/SOD/test.lst'
    elif sal_mode == 't':
        image_root = data_root + '/CarDD-TE/CarDD-TE-Image/'
        image_source = data_root + '/CarDD-TE/test.lst'
    elif sal_mode == 'm_r':  # for speed test
        image_root = data_root + '/data/MSRA/Imgs_resized/'
        image_source = data_root + '/data/MSRA/test_resized.lst'

    return image_root, image_source


def main(config):
    if config.mode == 'train':
        train_loader = get_loader(config)
        run = 0
        while os.path.exists("%s/run-%d" % (config.save_folder, run)):
            run += 1
        os.mkdir("%s/run-%d" % (config.save_folder, run))
        os.mkdir("%s/run-%d/models" % (config.save_folder, run))
        config.save_folder = "%s/run-%d" % (config.save_folder, run)
        train = Solver(train_loader, None, config)
        train.train()
    elif config.mode == 'test':
        test_loader = get_loader(config, mode='test')
        if not os.path.exists(config.test_fold): os.makedirs(config.test_fold)
        test = Solver(None, test_loader, config)
        test.test()
    else:
        raise IOError("illegal input!!!")


if __name__ == '__main__':
    root = r'/workspace/wangxinkuang'
    vgg_path = root + '/model/pretrained/vgg16_20M.pth'
    resnet_path = root + '/model/pretrained/resnet50_caffe.pth'

    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--n_color', type=int, default=3)
    parser.add_argument('--lr', type=float, default=5e-5)  # Learning rate resnet:5e-5, vgg:1e-4
    parser.add_argument('--wd', type=float, default=0.0005)  # Weight decay
    parser.add_argument('--no-cuda', dest='cuda', action='store_false')

    # Training settings
    parser.add_argument('--arch', type=str, default='resnet')  # resnet or vgg
    parser.add_argument('--pretrained_model', type=str, default=resnet_path)
    parser.add_argument('--epoch', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=1)  # only support 1 now
    parser.add_argument('--num_thread', type=int, default=1)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--save_folder', type=str, default='/workspace/wangxinkuang/model/saliency_detection/sgl_krn')
    parser.add_argument('--epoch_save', type=int, default=3)
    parser.add_argument('--iter_size', type=int, default=10)
    parser.add_argument('--show_every', type=int, default=100)

    parser.add_argument('--data_root', type=str, default='/workspace/wangxinkuang/data/saliency_detection/CarDD')
    # Train data
    parser.add_argument('--train_root', type=str, default=root + '/data/CarDD/CarDD-TR')
    parser.add_argument('--train_list', type=str, default=root + '/data/CarDD/CarDD-TR/train_pair.lst')

    # Testing settings
    parser.add_argument('--model', type=str, default=None)  # Snapshot
    parser.add_argument('--test_model', type=str, default=None)  # Snapshot
    parser.add_argument('--test_fold', type=str, default=None)  # Test results saving folder
    parser.add_argument('--sal_mode', type=str, default='e')  # Test image dataset

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    config = parser.parse_args()

    if not os.path.exists(config.save_folder):
        # print(os.getcwd())
        os.makedirs(config.save_folder)

    config.train_root = config.data_root + '/CarDD-TR'
    config.train_list = config.data_root + '/CarDD-TR/train_pair.lst'
    # Get test set info
    # test_root, test_list = get_test_info(config.sal_mode, config.data_root)
    # config.test_root = test_root
    # config.test_list = test_list
    config.test_root = config.data_root + '/CarDD-TE/CarDD-TE-Image/'
    config.test_list = config.data_root + '/CarDD-TE/test.lst'
    device = config.device
    torch.cuda.set_device(device)

    main(config)

