from torch.nn import  Module, Parameter
import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable

def where(cond, x_1, x_2):
    cond = cond.float()
    return (cond * x_1) + ((1 - cond) * x_2)


class ArcFullyConnected(Module):

    def __init__(self, in_features, out_features, s=64, m=0.5, is_pw=True, is_hard=True):
        super(ArcFullyConnected, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.is_pw = is_pw
        self.is_hard = is_hard
        assert s > 0
        assert 0 <= m < 0.5 * math.pi
        # print(s, m, is_pw, is_hard)
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def __repr__(self):
        return ('in_features={}, out_features={}, s={}, m={}'
                .format(self.in_features, self.out_features, self.s, self.m))

    def forward(self, embed, label):
        n_weight = F.normalize(self.weight, p=2, dim=1)
        n_embed = embed * self.s #F.normalize(embed, p=2, dim=1) * self.s
        out = F.linear(n_embed, n_weight)
        score = out.gather(1, label.view(-1, 1))
        cos_y = score / self.s
        sin_y = torch.sqrt(1 - cos_y ** 2)
        arc_score = self.s * (cos_y * math.cos(self.m) - sin_y * math.sin(self.m))
        if self.is_pw:
            if not self.is_hard:
                arc_score = where(score > 0, arc_score, score)
            else:
                mm = math.sin(math.pi - self.m) * self.m  # actually it is sin(m)*m
                th = math.cos(math.pi - self.m)  # actually it is -cos(m)
                arc_score = where((score - th) > 0, arc_score, score - self.s * mm)
        one_hot = Variable(torch.cuda.FloatTensor(out.shape).fill_(0))
        out += (arc_score - score) * one_hot.scatter_(1, label.view(-1, 1), 1)
        return out