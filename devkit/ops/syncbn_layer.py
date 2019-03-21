import torch
from torch.autograd import Function
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.distributed as dist
import torch.nn as nn

class SyncBNFunc(Function):

    @staticmethod
    def forward(ctx, in_data, scale_data, shift_data, running_mean, running_var, eps, momentum, training):
        if in_data.is_cuda:
            ctx.eps =eps
            N, C, H, W = in_data.size()
            in_data = in_data.view(N, C, -1)
            mean_in = in_data.mean(-1, keepdim=True)
            var_in = in_data.var(-1, keepdim=True)
            temp = var_in + mean_in ** 2
            if training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2

                sum_x = mean_bn ** 2 + var_bn
                dist.all_reduce(mean_bn)
                mean_bn /= dist.get_world_size()
                dist.all_reduce(sum_x)
                sum_x /= dist.get_world_size()
                var_bn = sum_x - mean_bn ** 2

                running_mean.mul_(momentum)
                running_mean.add_((1 - momentum) * mean_bn.data)
                running_var.mul_(momentum)
                running_var.add_((1 - momentum) * var_bn.data)

            else:
                mean_bn = torch.autograd.Variable(running_mean)
                var_bn = torch.autograd.Variable(running_var)

            x_hat = (in_data - mean_bn) / (var_bn+ ctx.eps).sqrt()
            x_hat = x_hat.view(N, C, H, W)
            out_data = x_hat * scale_data + shift_data

            ctx.save_for_backward(in_data.data, scale_data.data, x_hat.data,  mean_bn.data, var_bn.data)
        else:
            raise RuntimeError('SyncBNFunc only support CUDA computation!')
        return out_data

    @staticmethod
    def backward(ctx, grad_outdata):
        if grad_outdata.is_cuda:

            in_data, scale_data, x_hat, mean_bn, var_bn =  ctx.saved_tensors

            N, C, H, W = grad_outdata.size()
            scaleDiff = torch.sum(grad_outdata * x_hat,[0,2,3],keepdim=True)
            shiftDiff = torch.sum(grad_outdata,[0,2,3],keepdim=True)
            dist.all_reduce(scaleDiff)
            dist.all_reduce(shiftDiff)

            inDiff = scale_data / (var_bn.view(1,C,1,1) + ctx.eps).sqrt() *(grad_outdata - 1 / (N*H*W*dist.get_world_size()) * (scaleDiff * x_hat + shiftDiff))

        else:
            raise RuntimeError('SyncBNFunc only support CUDA computation!')
        return inDiff, scaleDiff, shiftDiff, None, None, None, None, None

class SyncBatchNorm2d(Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.9,last_gamma=False):
        super(SyncBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.last_gamma = last_gamma

        self.weight = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.bias = Parameter(torch.Tensor(1, num_features, 1, 1))

        self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
        self.register_buffer('running_var', torch.ones(1, num_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))

    def forward(self, in_data):
        return SyncBNFunc.apply(
                    in_data, self.weight, self.bias,  self.running_mean, self.running_var, self.eps, self.momentum, self.training)
