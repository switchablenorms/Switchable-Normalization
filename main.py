import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models import model_factory
import util
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-m','--model',  default='resnet50', help='model architecture name')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('-uma', '--using-moving-average', dest='using_moving_average', action='store_true',
                    help='in sn true is using moving average')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--model-dir','-md', default='save model path', type=str)
parser.add_argument('--checkpoint-path','-ckpt', default='checkpoint path', type=str)

best_prec1 = 0
ITER_COMPUTE_BATCH_AVEARGE = 200
class ColorAugmentation(object):
  def __init__(self, eig_vec=None, eig_val=None):
    if eig_vec == None:
      eig_vec = torch.Tensor([
        [0.4009, 0.7192, -0.5675],
        [-0.8140, -0.0045, -0.5808],
        [0.4203, -0.6948, -0.5836],
      ])
    if eig_val == None:
      eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])
    self.eig_val = eig_val
    self.eig_vec = eig_vec

  def __call__(self, tensor):
    assert tensor.size(0) == 3
    alpha = torch.normal(mean=torch.zeros_like(self.eig_val)) * 0.1
    quatity = torch.mm(self.eig_val * alpha, self.eig_vec)
    tensor = tensor + quatity.view(3, 1, 1)
    return tensor

def main():
    global args, best_prec1
    args = parser.parse_args()

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    print("=> creating model '{}'".format(args.model))
    if 'sn' in args.model:
        model = model_factory.get_model(args.model, using_moving_average = args.using_moving_average)
    else:
        model = model_factory.get_model(args.model)

    if not args.distributed:
        if args.model.startswith('alexnet') or args.model.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # auto resume from a checkpoint
    model_dir = args.model_dir
    start_epoch = 0
    if not os.path.exists(model_dir) :
        os.makedirs(model_dir)
    if args.evaluate:
        util.load_state_ckpt(args.checkpoint_path, model)
    else:
        best_prec1, start_epoch = util.load_state(model_dir, model, optimizer=optimizer)
    writer = SummaryWriter(model_dir)

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
      datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
      ])),
      batch_size=args.batch_size, shuffle=False,
      num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, 0, writer)
        return

    train_dataset_multi_scale = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ColorAugmentation(),
            normalize,
        ]))

    train_dataset = datasets.ImageFolder(
      traindir,
      transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ColorAugmentation(),
        normalize,
      ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader_multi_scale = torch.utils.data.DataLoader(
        train_dataset_multi_scale, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    train_loader = torch.utils.data.DataLoader(
      train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
      num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    if not args.using_moving_average:
        train_dataset_snhelper = datasets.ImageFolder(
          traindir,
          transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
          ]))
        train_loader_snhelper = torch.utils.data.DataLoader(
          train_dataset_snhelper, batch_size=args.batch_size * torch.cuda.device_count(), shuffle=(train_sampler is None),
          num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        if epoch < 95:
            train(train_loader_multi_scale, model, criterion, optimizer, epoch, writer)
        else:
            train(train_loader, model, criterion, optimizer, epoch, writer)

        if not args.using_moving_average:
            sn_helper(train_loader_snhelper, model)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch, writer)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        util.save_checkpoint(model_dir, {
          'epoch': epoch + 1,
          'model': args.model,
          'state_dict': model.state_dict(),
          'best_prec1': best_prec1,
          'optimizer': optimizer.state_dict(),
        }, is_best)

def train(train_loader, model, criterion, optimizer, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            niter = epoch * len(train_loader) + i
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], niter)
            writer.add_scalar('Train/Avg_Loss', losses.avg, niter)
            writer.add_scalar('Train/Avg_Top1', top1.avg / 100.0, niter)
            writer.add_scalar('Train/Avg_Top5', top5.avg / 100.0, niter)

def sn_helper(train_loader, model):

  model.train()

  for name, param in model.state_dict().items():
    if 'running_mean' in name:
      param.fill_(0)
    elif 'running_var' in name:
      param.fill_(0)

  with torch.no_grad():
    for i, (input, target) in enumerate(train_loader):
      if i == ITER_COMPUTE_BATCH_AVEARGE:
        break
      # target = target.cuda(non_blocking=True)
      model(input)

  for name, param in model.state_dict().items():
    if 'running_mean' in name:
      param /= ITER_COMPUTE_BATCH_AVEARGE
      model.state_dict()[name.replace('running_mean', 'running_var')] /= ITER_COMPUTE_BATCH_AVEARGE
      model.state_dict()[name.replace('running_mean', 'running_var')] -= param ** 2

def validate(val_loader, model, criterion, epoch, writer):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        niter = (epoch + 1)
        writer.add_scalar('Eval/Avg_Loss', losses.avg, niter)
        writer.add_scalar('Eval/Avg_Top1', top1.avg / 100.0, niter)
        writer.add_scalar('Eval/Avg_Top5', top5.avg / 100.0, niter)

    return top1.avg

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

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    main()
