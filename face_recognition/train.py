from __future__ import division
import argparse
import os
import time
import torch.distributed as dist
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import yaml
import sys
from tensorboardX import SummaryWriter
from models.model_builder import ArcFaceWithLoss
import os.path as osp
sys.path.append(osp.abspath(osp.join(__file__, '../../')))

from devkit.core import (init_dist, broadcast_params, average_gradients,
                         load_state, save_checkpoint, LRScheduler)

from devkit.dataset.facedataset import FaceDataset,  BigdataSampler

parser = argparse.ArgumentParser(
    description='Pytorch Imagenet Training')
parser.add_argument('--config', default='configs/config_resnetv1sn50.yaml')
parser.add_argument(
    '--port', default=29500, type=int, help='port of server')
parser.add_argument('--world-size', default=1, type=int)
parser.add_argument("--local_rank", type=int)
parser.add_argument('--rank', default=0, type=int)
parser.add_argument('--master', default='127.0.0.1', type=str)
parser.add_argument('--model_dir', type=str)
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
args = parser.parse_args()

def main():
    global args, best_prec1
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)


    rank, world_size = init_dist(
        backend='nccl', port=args.port)
    args.rank = rank
    args.world_size = world_size

    # create model

    model = ArcFaceWithLoss(args.backbone, args.class_num, args.norm_func, args.embedding_size, args.use_se)

    model.cuda()
    broadcast_params(model)


    optimizer = torch.optim.SGD(model.parameters(), args.base_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # auto resume from a checkpoint
    model_dir = args.model_dir
    start_epoch = 0
    if args.rank == 0 and not os.path.exists(model_dir):
        os.makedirs(model_dir)

    best_prec1, start_epoch = load_state(model_dir, model, optimizer=optimizer)
    if args.rank == 0:
        writer = SummaryWriter(model_dir)
    else:
        writer = None

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    train_dataset = FaceDataset(
        True,
        args,
        transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))


    train_sampler = BigdataSampler(
        train_dataset,
        num_sub_epochs=2,
        finegrain_factor=10000,
        seed=1000,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size//args.world_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, sampler=train_sampler)

    niters = len(train_loader)

    lr_scheduler = LRScheduler(optimizer, niters, args)

    for epoch in range(start_epoch, args.epochs):
        train(train_loader, model, optimizer, lr_scheduler, epoch, writer)
        if rank == 0:
            save_checkpoint(model_dir, {
                'epoch': epoch + 1,
                'model': args.backbone,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, False)


def train(train_loader, model, optimizer, lr_scheduler, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    world_size = args.world_size
    rank = args.rank

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        lr_scheduler.update(i, epoch)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input.cuda())
        target_var = torch.autograd.Variable(target)
        # compute output

        loss = model(input_var, target_var, extract_mode=False) / world_size
        reduced_loss = loss.data.clone()
        dist.all_reduce_multigpu([reduced_loss])
        losses.update(reduced_loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        average_gradients(model)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and rank == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            niter = epoch * len(train_loader) + i
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], niter)
            writer.add_scalar('Train/Avg_Loss', losses.avg, niter)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
