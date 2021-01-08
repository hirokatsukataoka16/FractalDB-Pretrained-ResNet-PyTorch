import math

import os
import numpy as np
import torch
import torch.nn as nn
from args import conf
args = conf()

__all__ = [ 'bn_AlexNet', 'bn_alexnet']

class bn_AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(bn_AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(96, eps=1e-05, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384, eps=1e-05, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384, eps=1e-05, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, args.numof_fclayer),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(args.numof_fclayer, args.numof_fclayer),
            nn.ReLU(inplace=True),
            nn.Linear(args.numof_fclayer, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256*6*6)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def bn_alexnet(pretrained=False, **kwargs):
    model = bn_AlexNet(**kwargs)
    return model


CFG = {
    '2012': [(96, 11, 4, 2), 'M', (256, 5, 1, 2), 'M', (384, 3, 1, 1), (384, 3, 1, 1), (256, 3, 1, 1), 'M']
}


class dc_AlexNet(nn.Module):
    def __init__(self, features, num_classes, sobel):
        super(dc_AlexNet, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Dropout(0.5),
                            nn.Linear(256 * 6 * 6, args.numof_fclayer),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(args.numof_fclayer, args.numof_fclayer),
                            nn.ReLU(inplace=True))

        self.top_layer = nn.Linear(args.numof_fclayer, num_classes)
        self._initialize_weights()

        if sobel:
            grayscale = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)
            grayscale.weight.data.fill_(1.0 / 3.0)
            grayscale.bias.data.zero_()
            sobel_filter = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
            sobel_filter.weight.data[0, 0].copy_(
                torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
            )
            sobel_filter.weight.data[1, 0].copy_(
                torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            )
            sobel_filter.bias.data.zero_()
            self.sobel = nn.Sequential(grayscale, sobel_filter)
            for p in self.sobel.parameters():
                p.requires_grad = False
        else:
            self.sobel = None

    def forward(self, x):
        if self.sobel:
            x = self.sobel(x)
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        if self.top_layer:
            x = self.top_layer(x)
        return x

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers_features(cfg, input_dim, bn):
    layers = []
    in_channels = input_dim
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v[0], kernel_size=v[1], stride=v[2], padding=v[3])
            if bn:
                layers += [conv2d, nn.BatchNorm2d(v[0]), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v[0]
    return nn.Sequential(*layers)


#def bn_alex_deepclustering(sobel=False, bn=True, out=1000):
#    dim = 2 + int(not sobel)
#    model = dc_AlexNet(make_layers_features(CFG['2012'], dim, bn=bn), out, sobel)
#    return model

#class rot_Flatten(nn.Module):
#    def __init__(self):
#        super(rot_Flatten, self).__init__()
#
#    def forward(self, feat):
#        return feat.view(feat.size(0), -1)

#class rot_AlexNet(nn.Module):
#    def __init__(self, opt):
#        super(rot_AlexNet, self).__init__()
#        num_classes = opt['num_classes']
#
#        conv1 = nn.Sequential(
#            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
#            nn.BatchNorm2d(64),
#            nn.ReLU(inplace=True),
#        )
#        pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
#        conv2 = nn.Sequential(
#            nn.Conv2d(64, 192, kernel_size=5, padding=2),
#            nn.BatchNorm2d(192),
#            nn.ReLU(inplace=True),
#        )
#        pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
#        conv3 = nn.Sequential(
#            nn.Conv2d(192, 384, kernel_size=3, padding=1),
#            nn.BatchNorm2d(384),
#            nn.ReLU(inplace=True),
#        )
#        conv4 = nn.Sequential(
#            nn.Conv2d(384, 256, kernel_size=3, padding=1),
#            nn.BatchNorm2d(256),
#            nn.ReLU(inplace=True),
#        )
#        conv5 = nn.Sequential(
#            nn.Conv2d(256, 256, kernel_size=3, padding=1),
#            nn.BatchNorm2d(256),
#            nn.ReLU(inplace=True),
#        )
#        pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
#
#        num_pool5_feats = 6 * 6 * 256
#        fc_block = nn.Sequential(
#            rot_Flatten(),
#            nn.Linear(num_pool5_feats, args.numof_fclayer, bias=False),
#            nn.BatchNorm1d(args.numof_fclayer),
#            nn.ReLU(inplace=True),
#            nn.Linear(args.numof_fclayer, args.numof_fclayer, bias=False),
#            nn.BatchNorm1d(args.numof_fclayer),
#            nn.ReLU(inplace=True),
#        )
#        classifier = nn.Sequential(
#            nn.Linear(args.numof_fclayer, num_classes),
#        )
#
#        self._feature_blocks = nn.ModuleList([
#            conv1, pool1, conv2, pool2, conv3, 
#            conv4, conv5, pool5, fc_block, classifier,
#        ])
#        self.all_feat_names = [
#            'conv1', 'pool1', 'conv2', 'pool2', 'conv3',
#            'conv4', 'conv5', 'pool5', 'fc_block', 'classifier',
#        ]
#        assert(len(self.all_feat_names) == len(self._feature_blocks))
#
#    def _parse_out_keys_arg(self, out_feat_keys):
#
#        # By default return the features of the last layer / module.
#        out_feat_keys = [self.all_feat_names[-1],] if out_feat_keys is None else out_feat_keys
#
#        if len(out_feat_keys) == 0:
#            raise ValueError('Empty list of output feature keys.')
#        for f, key in enumerate(out_feat_keys):
#            if key not in self.all_feat_names:
#                raise ValueError('Feature with name {0} does not exist. Existing features: {1}.'.format(key, self.all_feat_names))
#            elif key in out_feat_keys[:f]:
#                raise ValueError('Duplicate output feature key: {0}.'.format(key))
#
#        # Find the highest output feature in `out_feat_keys
#        max_out_feat = max([self.all_feat_names.index(key) for key in out_feat_keys])
#
#        return out_feat_keys, max_out_feat
#
#    def forward(self, x, out_feat_keys=None):
#        """Forward an image `x` through the network and return the asked output features.
#
#        Args:
#          x: input image.
#          out_feat_keys: a list/tuple with the feature names of the features
#                that the function should return. By default the last feature of
#                the network is returned.
#
#        Return:
#            out_feats: If multiple output features were asked then `out_feats`
#                is a list with the asked output features placed in the same
#                order as in `out_feat_keys`. If a single output feature was
#                asked then `out_feats` is that output feature (and not a list).
#        """
#        out_feat_keys, max_out_feat = self._parse_out_keys_arg(out_feat_keys)
#        out_feats = [None] * len(out_feat_keys)
#
#        feat = x
#        for f in range(max_out_feat+1):
#            feat = self._feature_blocks[f](feat)
#            key = self.all_feat_names[f]
#            if key in out_feat_keys:
#                out_feats[out_feat_keys.index(key)] = feat
#
#        out_feats = out_feats[0] if len(out_feats)==1 else out_feats
#        return out_feats
#
#    def get_L1filters(self):
#        convlayer = self._feature_blocks[0][0]
#        batchnorm = self._feature_blocks[0][1]
#        filters = convlayer.weight.data
#        scalars = (batchnorm.weight.data / torch.sqrt(batchnorm.running_var + 1e-05))
#        filters = (filters * scalars.view(-1, 1, 1, 1).expand_as(filters)).cpu().clone()
#
#        return filters
#
#def load_pretrained(network, pretrained_path):
#    assert(os.path.isfile(pretrained_path))
#    pretrained_model = torch.load(pretrained_path)
#    if pretrained_model['network'].keys() == network.state_dict().keys():
#        network.load_state_dict(pretrained_model['network'])
#    else:
#        for pname, param in network.named_parameters():
#            if pname in pretrained_model['network']:
#                param.data.copy_(pretrained_model['network'][pname])
