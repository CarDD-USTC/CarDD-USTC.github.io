import torch
from torch import nn
import torch.nn.functional as F
from networks.deeplab_resnet import \
    resnet50_locate
from networks.vgg import vgg16_locate

config_vgg = {'convert': [[128, 256, 512, 512, 512], [64, 128, 256, 512, 512]],
              'deep_pool': [[512, 512, 256, 128], [512, 256, 128, 128], [True, True, True, False],
                            [True, True, True, False]], 'score': 128}  # no convert layer, no conv6

config_resnet = {'convert': [[64, 256, 512, 1024, 2048], [128, 256, 256, 512, 512]],
                 'deep_pool': [[512, 512, 256, 256, 128], [512, 256, 256, 128, 128], [False, True, True, True, False],
                               [True, True, True, True, False]], 'score': 128}


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
        y = x
        for i in range(len(self.pools_sizes)):
            y = self.convs[i](self.pools[i](y))
            resl = torch.add(resl, F.interpolate(y, x_size[2:], mode='bilinear', align_corners=True))
        resl = self.relu(resl)
        if self.need_x2:
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


class KRN(nn.Module):
    def __init__(self, base_model_cfg, base, convert_layers, score_layers):
        super(KRN, self).__init__()
        self.base_model_cfg = base_model_cfg
        self.base = base  # 基本网络是一样的

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
            conv2merge = self.convert(conv2merge)
        conv2merge = conv2merge[::-1]

        merge_solid1 = self.DeepPool_solid1(conv2merge[0], conv2merge[1])
        out_merge_solid1 = self.score_solid1(merge_solid1, x_size)
        out_merge_solid1 = F.sigmoid(out_merge_solid1)
        fea_reduce1 = self.conv_reduce1(merge_solid1)
        fea_reduce1 = self.relu(fea_reduce1)

        merge_solid2 = self.DeepPool_solid2(merge_solid1, conv2merge[2])
        out_merge_solid2 = self.score_solid2(merge_solid2, x_size)
        out_merge_solid2 = F.sigmoid(out_merge_solid2)
        fea_reduce2 = self.conv_reduce2(merge_solid2)
        fea_reduce2 = self.relu(fea_reduce2)

        merge_solid3 = self.DeepPool_solid3(merge_solid2, conv2merge[3])
        out_merge_solid3 = self.score_solid3(merge_solid3, x_size)
        out_merge_solid3 = F.sigmoid(out_merge_solid3)
        fea_reduce3 = self.conv_reduce3(merge_solid3)
        fea_reduce3 = self.relu(fea_reduce3)

        merge_solid4 = self.DeepPool_solid4(merge_solid3, conv2merge[4])
        out_merge_solid4 = self.score_solid4(merge_solid4, x_size)
        out_merge_solid4 = F.sigmoid(out_merge_solid4)
        fea_reduce4 = merge_solid4

        merge_solid5 = self.DeepPool_solid5(merge_solid4)
        merge_solid = self.score_solid(merge_solid5, x_size)
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
        feasum_out = self.score_sum_out(feasum_out, x_size)
        feasum_out = F.sigmoid(feasum_out)

        return feasum_out, merge_solid, out_merge_solid1, out_merge_solid2, out_merge_solid3, out_merge_solid4


def build_model(base_model_cfg='vgg'):
    if base_model_cfg == 'vgg':
        return KRN(base_model_cfg, *extra_layer(base_model_cfg, vgg16_locate()))
    elif base_model_cfg == 'resnet':
        return KRN(base_model_cfg, *extra_layer(base_model_cfg, resnet50_locate()))


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()
