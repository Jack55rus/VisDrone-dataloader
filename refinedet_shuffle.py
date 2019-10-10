import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *
from data import voc_refinedet, coco_refinedet, visdrone_refinedet
import os
import time

from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import math

import shutil
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from ShuffleNetV2 import shufflenetv2


class RefineDet(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, ARM, ODM, TCB, num_classes):
        super(RefineDet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = visdrone_refinedet # debug
        # print(self.cfg[str(size)]) # debugging
        # self.cfg = (coco_refinedet, voc_refinedet, visdrone_refinedet)[num_classes == 13]
        # print(self.cfg[str(size)]) # debugging
        self.priorbox = PriorBox(self.cfg[str(size)])
        with torch.no_grad():
            self.priors = self.priorbox.forward()
       
        self.size = size

        # SSD network
        self.shuffle = base
        self.extra_conv_1 = nn.Conv2d(464, 928, kernel_size=3, stride=2, padding=1)
        self.deconv_st3 = nn.ConvTranspose2d(232, 232, 2, 2)
        self.deconv_st4 = nn.ConvTranspose2d(464, 464, 2, 2)
        self.deconv_extra = nn.ConvTranspose2d(928, 928, 2, 2)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.stage2_L2Norm = L2Norm(116, 10)
        self.stage3_L2Norm = L2Norm(232, 8)
        self.arm_loc = nn.ModuleList(ARM[0])
        self.arm_conf = nn.ModuleList(ARM[1])
        self.odm_loc = nn.ModuleList(ODM[0])
        self.odm_conf = nn.ModuleList(ODM[1])
        #self.tcb = nn.ModuleList(TCB)
        self.tcb0 = nn.ModuleList(TCB[0])
        self.tcb1 = nn.ModuleList(TCB[1])
        self.tcb2 = nn.ModuleList(TCB[2])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect_RefineDet(num_classes, self.size, 0, 1000, 0.01, 0.45, 0.01, 500) # was 1k at the 4th pos

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        # forward_start = time.time()
        sources = list()
        
        tcb_source = list()
        arm_loc = list()
        arm_conf = list()
        odm_loc = list()
        odm_conf = list()
        # print('self.priors shape = ', self.priors.shape) # debug
        # print(self.shuffle)
        
        # sources = [self.shuffle(x)[1], self.shuffle(x)[2], self.shuffle(x)[3]]
        sources = [self.shuffle(x)[1], self.shuffle(x)[2]] # 48x48x232 / 24x24x464
        
        extra_apply = self.extra_conv_1(sources[1]) # 12x12x928
        extra_apply = F.relu(extra_apply, inplace=True)
        sources.append(extra_apply) # sources: 48x48x232 / 24x24x464 / 12x12x928
        
        sources[0] = self.deconv_st3(sources[0]) # sources[0]: 96x96x232
        sources[1] = self.deconv_st4(sources[1]) # sources[1]: 48x48x464
        sources[2] = self.deconv_extra(sources[2]) # sources[2]: 24x24x928
        """
        extra_conv_1 = nn.Conv2d(1024, 256, kernel_size=1)
        extra_conv_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        extra_apply = extra_conv_1(sources[2])
        extra_apply = extra_conv_2(extra_apply)
        extra_apply = F.relu(extra_apply, inplace=True)
        sources.append(extra_apply)
        """
        # print('self.shuffle(x)[1].shape = ', self.shuffle(x)[1].shape)
        # print('self.shuffle(x)[2].shape = ', self.shuffle(x)[2].shape)
        # apply ARM and ODM to source layers
        for (x, l, c) in zip(sources, self.arm_loc, self.arm_conf):
            arm_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            arm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        arm_loc = torch.cat([o.view(o.size(0), -1) for o in arm_loc], 1)
        arm_conf = torch.cat([o.view(o.size(0), -1) for o in arm_conf], 1)
        #print([x.size() for x in sources])
        # calculate TCB features
        #print([x.size() for x in sources])
        p = None
        for k, v in enumerate(sources[::-1]):
            s = v
            # print('s.shape = ', s.shape)
            for i in range(3):
                s = self.tcb0[(2-k)*3 + i](s)
                #print(s.size())
            if k != 0:
                # print('entered tcb1; k = ', k)
                u = p
                # print('s shape = ', s.shape)
                # print('p shape = ', p.shape)
                u = self.tcb1[2-k](u)
                # print('u shape = ', u.shape)
                s += u
            for i in range(3):
                s = self.tcb2[(2-k)*3 + i](s)
            p = s
            tcb_source.append(s)
        #print([x.size() for x in tcb_source])
        tcb_source.reverse()

        # apply ODM to source layers
        for (x, l, c) in zip(tcb_source, self.odm_loc, self.odm_conf):
            odm_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            odm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        odm_loc = torch.cat([o.view(o.size(0), -1) for o in odm_loc], 1)
        odm_conf = torch.cat([o.view(o.size(0), -1) for o in odm_conf], 1)
        #print(arm_loc.size(), arm_conf.size(), odm_loc.size(), odm_conf.size())
        # print(self.priors.type(type(x.data))) # debug 
        # print(self.priors.shape) # debug
        # det_time = 0
        if self.phase == "test":
            # print('arm_loc shape ', arm_loc.shape)
            # print('arm_conf shape ', arm_conf.shape)
            # print('odm_loc shape ', odm_loc.shape)
            # print('odm_conf shape ', odm_conf.shape)
            #print(loc, conf)
            # detect_start = time.time()
            output = self.detect(
                arm_loc.view(arm_loc.size(0), -1, 4),           # arm loc preds
                self.softmax(arm_conf.view(arm_conf.size(0), -1,
                             2)),                               # arm conf preds
                odm_loc.view(odm_loc.size(0), -1, 4),           # odm loc preds
                self.softmax(odm_conf.view(odm_conf.size(0), -1,
                             self.num_classes)),                # odm conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
            # detect_end = time.time()
            # det_time = detect_end - detect_start
        else:
            output = (
                arm_loc.view(arm_loc.size(0), -1, 4),
                arm_conf.view(arm_conf.size(0), -1, 2),
                odm_loc.view(odm_loc.size(0), -1, 4),
                odm_conf.view(odm_conf.size(0), -1, self.num_classes),
                self.priors
            )
        # forward_end = time.time()
        # forward_time = forward_end - forward_start - det_time
        # print(forward_time)
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage), strict=False)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

def arm_multibox_shuffle():
    arm_loc_layers = []
    arm_conf_layers = []
    in_channels = [232, 464, 928]
    for i in range(len(in_channels)):
        arm_loc_layers += [nn.Conv2d(in_channels[i],
                                 12, kernel_size=3, padding=1)]
        arm_conf_layers += [nn.Conv2d(in_channels[i],
                        6, kernel_size=3, padding=1)]
    return (arm_loc_layers, arm_conf_layers)

def odm_multibox_shuffle(num_classes):
    odm_loc_layers = []
    odm_conf_layers = []
    num_odm_ch = 3
    for i in range(num_odm_ch):
        odm_loc_layers += [nn.Conv2d(256, 12, kernel_size=3, padding=1)]
        odm_conf_layers += [nn.Conv2d(256, 3 * num_classes, kernel_size=3, padding=1)]
    return (odm_loc_layers, odm_conf_layers)

def add_tcb_shuffle():
    feature_scale_layers = []
    feature_upsample_layers = []
    feature_pred_layers = []
    feeding_ch = [232, 464, 928]
    for i in range(len(feeding_ch)):
        feature_scale_layers += [nn.Conv2d(feeding_ch[i], 256, 3, padding=1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(256, 256, 3, padding=1)
        ]
        feature_pred_layers += [nn.ReLU(inplace=True),
                                nn.Conv2d(256, 256, 3, padding=1),
                                nn.ReLU(inplace=True)
        ]
        if i != len(feeding_ch) - 1:
            feature_upsample_layers += [nn.ConvTranspose2d(256, 256, 2, 2)]
    return (feature_scale_layers, feature_upsample_layers, feature_pred_layers)


###################################################################################################

def build_refinedet(phase, size=320, num_classes=13):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 320 and size != 512 and size != 768 and size != 960:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only RefineDet320 and RefineDet512 is supported!")
        return 
    base_ = shufflenetv2(1.)
    # extras_ = add_extras(extras[str(size)], size, 1024)
    ARM_ = arm_multibox_shuffle()
    print('passed # classes = ', num_classes)
    ODM_ = odm_multibox_shuffle(num_classes)
    TCB_ = add_tcb_shuffle()
    print('size = ', size) # debug
    return RefineDet(phase, size, base_, ARM_, ODM_, TCB_, num_classes)
