from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import RefineDetMultiBoxLoss
#from ssd import build_ssd
from models.refinedet import build_refinedet
# from data import VISDRONEDetection
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
from utils.logging import Logger

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VISDRONE', choices=['VOC', 'COCO', 'VISDRONE'],
                    type=str, help='VOC, COCO or VISDRONE')
parser.add_argument('--input_size', default='320', choices=['320', '512'],
                    type=str, help='RefineDet320 or RefineDet512')
parser.add_argument('--basenet', default='./weights/vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

sys.stdout = Logger(os.path.join(args.save_folder, 'log.txt'))

def train():
    if args.dataset == 'COCO':
        '''if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))'''
    elif args.dataset == 'VISDRONE':
        '''if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')'''
        cfg = visdrone_refinedet[args.input_size]
        dataset = VISDRONEDetection(transform=SSDAugmentation(cfg['min_dim'], MEANS))

    if args.visdom:
        import visdom
        viz = visdom.Visdom()

    refinedet_net = build_refinedet('train', cfg['min_dim'], cfg['num_classes'])
    net = refinedet_net
    print(net)
    #input()

    if args.cuda:
        net = torch.nn.DataParallel(refinedet_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        refinedet_net.load_weights(args.resume)
    else:
        #vgg_weights = torch.load(args.save_folder + args.basenet)
        vgg_weights = torch.load(args.basenet)
        print('Loading base network...')
        refinedet_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        refinedet_net.extras.apply(weights_init)
        refinedet_net.arm_loc.apply(weights_init)
        refinedet_net.arm_conf.apply(weights_init)
        refinedet_net.odm_loc.apply(weights_init)
        refinedet_net.odm_conf.apply(weights_init)
        #refinedet_net.tcb.apply(weights_init)
        refinedet_net.tcb0.apply(weights_init)
        refinedet_net.tcb1.apply(weights_init)
        refinedet_net.tcb2.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    arm_criterion = RefineDetMultiBoxLoss(2, 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)
    odm_criterion = RefineDetMultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda, use_ARM=True)

    net.train()
    # loss counters
    arm_loc_loss = 0
    arm_conf_loss = 0
    odm_loc_loss = 0
    odm_conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training RefineDet on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    if args.visdom:
        vis_title = 'RefineDet.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    arm_cumul = 0
    odm_cumul = 0
    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter']):
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            update_vis_plot(epoch, arm_loc_loss, arm_conf_loss, epoch_plot, None,
                            'append', epoch_size)
            # reset epoch loss counters
            arm_loc_loss = 0
            arm_conf_loss = 0
            odm_loc_loss = 0
            odm_conf_loss = 0
            epoch += 1

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        if args.cuda:
            images = images.cuda()
            targets = [ann.cuda() for ann in targets]
        else:
            images = images
            targets = [ann for ann in targets]
        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        arm_loss_l, arm_loss_c = arm_criterion(out, targets)
        odm_loss_l, odm_loss_c = odm_criterion(out, targets)
        #input()
        arm_loss = arm_loss_l + arm_loss_c
        odm_loss = odm_loss_l + odm_loss_c
        loss = arm_loss + odm_loss
        loss.backward()
        optimizer.step()
        t1 = time.time()
        arm_loc_loss += arm_loss_l.item()
        arm_conf_loss += arm_loss_c.item()
        odm_loc_loss += odm_loss_l.item()
        odm_conf_loss += odm_loss_c.item()
        period = 1000
        arm_cumul += (arm_loss_l.item() + arm_loss_c.item())
        odm_cumul += (odm_loss_l.item() + odm_loss_c.item())

        if iteration % period == 0 and iteration != 0:
            mean_arm_loss = arm_cumul/period
            mean_odm_loss = odm_cumul/period
            mean_total_loss = mean_arm_loss + mean_odm_loss
            print('Cumul ARM loss = %.4f' % mean_arm_loss, 'Cumul ODM loss = %.4f' % mean_odm_loss, 'total loss = %.4f' % mean_total_loss , sep = ' | ')
            arm_cumul = 0
            odm_cumul = 0


        if iteration % 100 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || ARM_L Loss: %.4f ARM_C Loss: %.4f ODM_L Loss: %.4f ODM_C Loss: %.4f ||' \
            % (arm_loss_l.item(), arm_loss_c.item(), odm_loss_l.item(), odm_loss_c.item()), end=' ')

        if args.visdom:
            update_vis_plot(iteration, arm_loss_l.data[0], arm_loss_c.data[0],
                            iter_plot, epoch_plot, 'append')

        if iteration != 0 and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(refinedet_net.state_dict(), args.save_folder 
            + '/RefineDet{}_{}_{}.pth'.format(args.input_size, args.dataset, 
            repr(iteration)))
    torch.save(refinedet_net.state_dict(), args.save_folder 
            + '/RefineDet{}_{}_final.pth'.format(args.input_size, args.dataset))


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    train()