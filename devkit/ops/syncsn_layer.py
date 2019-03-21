import torch
from torch.autograd import Function
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.distributed as dist
import torch.nn as nn

class SyncSNFunc(Function):

    @staticmethod
    def forward(ctx, in_data, scale_data, shift_data, mean_weight, var_weight,running_mean, running_var, eps, momentum, training):
        if in_data.is_cuda:
            ctx.eps =eps
            N, C, H, W = in_data.size()
            in_data = in_data.view(N, C, -1)
            mean_in = in_data.mean(-1, keepdim=True)
            var_in = in_data.var(-1, keepdim=True)

            mean_ln = mean_in.mean(1, keepdim=True)
            temp = var_in + mean_in ** 2
            var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

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

            softmax = nn.Softmax(0)
            mean_weight = softmax(mean_weight)
            var_weight = softmax(var_weight)

            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn

            x_hat = (in_data - mean) / (var + ctx.eps).sqrt()
            x_hat = x_hat.view(N, C, H, W)
            out_data = x_hat * scale_data + shift_data

            ctx.save_for_backward(in_data.data, scale_data.data, x_hat.data, mean.data, var.data, mean_in.data, var_in.data,
                                  mean_ln.data, var_ln.data, mean_bn.data, var_bn.data, mean_weight.data, var_weight.data)
        else:
            raise RuntimeError('SyncSNFunc only support CUDA computation!')
        return out_data

    @staticmethod
    def backward(ctx, grad_outdata):
        if grad_outdata.is_cuda:

            in_data, scale_data, x_hat, mean, var, mean_in, var_in, mean_ln, var_ln, mean_bn, var_bn, \
                mean_weight, var_weight=  ctx.saved_tensors

            N, C, H, W = grad_outdata.size()
            scaleDiff = torch.sum(grad_outdata * x_hat,[0,2,3],keepdim=True)
            shiftDiff = torch.sum(grad_outdata,[0,2,3],keepdim=True)
            # dist.all_reduce(scaleDiff)
            # dist.all_reduce(shiftDiff)
            x_hatDiff = scale_data * grad_outdata

            meanDiff = -1 / (var.view(N,C) + ctx.eps).sqrt() * torch.sum(x_hatDiff,[2,3])

            varDiff = -0.5 / (var.view(N,C) + ctx.eps) * torch.sum(x_hatDiff * x_hat,[2,3])

            term1 = grad_outdata * scale_data / (var.view(N,C,1,1) + ctx.eps).sqrt()

            term21 = var_weight[0] * 2 * (in_data.view(N,C,H,W) - mean_in.view(N,C,1,1)) / (H*W) * varDiff.view(N,C,1,1)
            term22 = var_weight[1] * 2 * (in_data.view(N,C,H,W) - mean_ln.view(N,1,1,1)) / (C*H*W) * torch.sum(varDiff,[1]).view(N,1,1,1)
            term23_tmp = torch.sum(varDiff,[0]).view(1,C,1,1)
            dist.all_reduce(term23_tmp)
            term23 = var_weight[2] * 2 * (in_data.view(N,C,H,W) - mean_bn.view(1,C,1,1)) / (N*H*W) * term23_tmp / dist.get_world_size()

            term31 = mean_weight[0] * meanDiff.view(N,C,1,1) / H / W
            term32 = mean_weight[1] * torch.sum(meanDiff,[1]).view(N,1,1,1) / C  / H / W
            term33_tmp = torch.sum(meanDiff,[0]).view(1,C,1,1)
            dist.all_reduce(term33_tmp)
            term33 = mean_weight[2] * term33_tmp / N  / H / W / dist.get_world_size()

            inDiff =term1 + term21 + term22 + term23 + term31 + term32 + term33

            mw1_diff = torch.sum(meanDiff * mean_in.view(N,C))
            mw2_diff = torch.sum(meanDiff * mean_ln.view(N, 1))
            mw3_diff = torch.sum(meanDiff * mean_bn.view(1, C))

            dist.all_reduce(mw1_diff)
            # mw1_diff /= dist.get_world_size()
            dist.all_reduce(mw2_diff)
            # mw2_diff /= dist.get_world_size()
            dist.all_reduce(mw3_diff)
            # mw3_diff /= dist.get_world_size()

            vw1_diff = torch.sum(varDiff * var_in.view(N, C))
            vw2_diff = torch.sum(varDiff * var_ln.view(N, 1))
            vw3_diff = torch.sum(varDiff * var_bn.view(1, C))

            dist.all_reduce(vw1_diff)
            # vw1_diff /= dist.get_world_size()
            dist.all_reduce(vw2_diff)
            # vw2_diff /= dist.get_world_size()
            dist.all_reduce(vw3_diff)
            # vw3_diff /= dist.get_world_size()

            mean_weight_Diff = mean_weight
            var_weight_Diff = var_weight

            mean_weight_Diff[0] = mean_weight[0] * (mw1_diff - mean_weight[0] * mw1_diff - mean_weight[1] * mw2_diff- mean_weight[2] * mw3_diff )
            mean_weight_Diff[1] = mean_weight[1] * (mw2_diff - mean_weight[0] * mw1_diff - mean_weight[1] * mw2_diff - mean_weight[2] * mw3_diff)
            mean_weight_Diff[2] = mean_weight[2] * (mw3_diff - mean_weight[0] * mw1_diff - mean_weight[1] * mw2_diff - mean_weight[2] * mw3_diff)
            var_weight_Diff[0] = var_weight[0] * (vw1_diff - var_weight[0] * vw1_diff - var_weight[1] * vw2_diff - var_weight[2] * vw3_diff)
            var_weight_Diff[1] = var_weight[1] * (vw2_diff - var_weight[0] * vw1_diff - var_weight[1] * vw2_diff - var_weight[2] * vw3_diff)
            var_weight_Diff[2] = var_weight[2] * (vw3_diff - var_weight[0] * vw1_diff - var_weight[1] * vw2_diff - var_weight[2] * vw3_diff)


        else:
            raise RuntimeError('SyncBNFunc only support CUDA computation!')
        return inDiff, scaleDiff, shiftDiff, mean_weight_Diff, var_weight_Diff, None, None, None, None, None

class SyncSwitchableNorm2d(Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.9,last_gamma=False):
        super(SyncSwitchableNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.last_gamma = last_gamma

        self.weight = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.bias = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.mean_weight = Parameter(torch.ones(3))
        self.var_weight = Parameter(torch.ones(3))

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
        return SyncSNFunc.apply(
                    in_data, self.weight, self.bias, self.mean_weight, self.var_weight, self.running_mean, self.running_var, self.eps, self.momentum, self.training)
