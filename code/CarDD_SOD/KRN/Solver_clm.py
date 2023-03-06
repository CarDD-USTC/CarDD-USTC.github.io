
import torch
from torch.nn import utils, functional as F
import os
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np
import cv2
import time
from KRN import *



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

class Mul_loss(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(Mul_loss, self).__init__()
        self.eps = 1e-8

    def forward(self, x, y, gt):
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        CCloss = torch.sum(vx * vy) / ((torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))) + self.eps)

        x_map_norm = (x - torch.mean(x)) / (torch.std(x) + self.eps)
        y_map_norm = (y - torch.mean(y)) / (torch.std(y) + self.eps)
        diff = torch.abs(x_map_norm - y_map_norm)
        m = torch.sum(torch.mul(diff, gt))
        # print(m)
        num = torch.sum(gt) + self.eps  # 求1的个数
        NSSloss = torch.div(m, num)

        #BCEloss = F.binary_cross_entropy(x, y, reduction='mean')
        max_x = torch.max(x)
        x = x / max_x
        sum_x = torch.sum(x)
        sum_y = torch.sum(y)
        x = x / (sum_x + self.eps)
        y = y / (sum_y + self.eps)
        KLDloss = torch.sum(y * torch.log(self.eps + y / (x + self.eps)))
        return 1 - CCloss + NSSloss + KLDloss

mulloss = Mul_loss()


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
            print('Loading pre-trained model from %s...' % self.config.model)
            if self.config.cuda:
                self.net.load_state_dict(torch.load('final.pth'))
            else:
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


    def test(self):
        mode_name = 'sal_fuse'
        time_s = time.time()
        img_num = len(self.test_loader)
        for i, data_batch in enumerate(self.test_loader):
            images, name, im_size = data_batch['image'], data_batch['name'][0], np.asarray(data_batch['size'])
            with torch.no_grad():
                images = Variable(images)
                if self.config.cuda:
                    images = images.cuda()
                feasum_out, merge_solid,  out_merge_solid1, out_merge_solid2, out_merge_solid3, out_merge_solid4 = self.net(
                    images)

                pred = np.squeeze(feasum_out).cpu().data.numpy()
                multi_fuse = 255 * pred
                cv2.imwrite(os.path.join(self.config.test_fold, name[:-4] + '_' + mode_name + '.png'), multi_fuse)
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
                sal_image, sal_label, sal_edge, sal_saliency = data_batch['sal_image'], data_batch['sal_label'], data_batch[
                    'sal_edge'], data_batch['sal_saliency']
                if (sal_image.size(2) != sal_label.size(2)) or (sal_image.size(3) != sal_label.size(3)):
                    print('IMAGE ERROR, PASSING```')
                    continue
                sal_image, sal_label, sal_edge, sal_saliency = Variable(sal_image), Variable(sal_label), Variable(sal_edge), Variable(sal_saliency)
                if self.config.cuda:
                    # cudnn.benchmark = True
                    sal_image, sal_label, sal_edge, sal_saliency = sal_image.cuda(), sal_label.cuda(), sal_edge.cuda(), sal_saliency.cuda()

                feasum_out, merge_solid,  out_merge_solid1, out_merge_solid2, out_merge_solid3, out_merge_solid4 = self.net(
                    sal_image)

                high_score = torch.trunc(sal_saliency, out=None)

                feasum_out_loss = mulloss(feasum_out, sal_saliency, high_score)
                solid_loss = mulloss(merge_solid, sal_saliency, high_score)
                solid_loss1 = mulloss(out_merge_solid1, sal_saliency, high_score)
                solid_loss2 = mulloss(out_merge_solid2, sal_saliency, high_score)
                solid_loss3 = mulloss(out_merge_solid3, sal_saliency, high_score)
                solid_loss4 = mulloss(out_merge_solid4, sal_saliency, high_score)


                sal_loss = (2*feasum_out_loss + solid_loss + solid_loss1 + solid_loss2 + solid_loss3 + solid_loss4) / (
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
                    print('Learning rate: ' + str(self.lr))
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