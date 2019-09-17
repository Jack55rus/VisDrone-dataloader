"""VISDRONE Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
Modified for the VisDrone dataset by Evgeny Markin

"""
from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import os


VISDRONE_CLASSES = (  'ignored',
    'pedestrian', 'people', 'bicycle', 'car',
    'van', 'truck', 'tricycle', 'awningtri', 'bus',
    'motorb', 'others')

# note: if you used our download scripts, this should be right
VISDRONE_ROOT = osp.join(HOME, "data/VISDRONE/")


class VISDRONEAnnotationTransform(object):
    """Transforms a VISDRONE annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, keep_difficult=False):
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class ind]
        """
        res = []
        with open(target,  'r') as f:
            f1 = f.readlines()
            for l in f1:
                y = l.split(',')
                if int(y[5]) == 0:
                    continue
                bndbox = []
                bndbox += [int(y[0])/width, int(y[1])/height, (int(y[0])+int(y[2]))/width, (int(y[1])+int(y[3]))/height, int(y[5])]
                res += [bndbox]


        #f = open(target, 'r')
        #f1 = f.readlines()
        #for l in f1:
            #y = l.split(',')
            #difficult = int(y[4]) == 1
            #if not self.keep_difficult and difficult:
                #continue

            #bndbox = []
            #bndbox += [int(y[0])/width, int(y[1])/height, (int(y[0])+int(y[2]))/width, (int(y[1])+int(y[3]))/height, int(y[5])]
            #res += [bndbox]
        #f.close()
        # print(res)
        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VISDRONEDetection(data.Dataset):
    """VisDrone Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VISDRONE folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self,
                 image_set='train',
                 transform=None, target_transform=VISDRONEAnnotationTransform(),
                 dataset_name='VOC0712'):
        # self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join(VISDRONE_ROOT, 'VisDrone2019_%s', 'annotations') % (self.image_set)
        self._imgpath = osp.join(VISDRONE_ROOT, 'VisDrone2019_%s', 'images') % (self.image_set)
        self.ids = list()
        for filename in os.listdir(self._annopath):
            self.ids.append(filename[0:-4])
        self._annopath = osp.join(self._annopath, '%s.txt')
        self._imgpath = osp.join(self._imgpath, '%s.jpg')
        # for (year, name) in image_sets:
            # rootpath = osp.join(self.root, 'VOC' + year)
            # for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                # self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        # print(img_id)
        # target = ET.parse(self._annopath % img_id).getroot()
        target = self._annopath % img_id
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
        # print(target, width, height, sep='|')

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        # anno = ET.parse(self._annopath % img_id).getroot()
        anno = self._annopath % img_id
        gt = self.target_transform(anno, 1, 1)
        return img_id, gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
