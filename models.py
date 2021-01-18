from layers import GraphConvolution, PairNorm
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch


class FastGAE(nn.Module):
    def __init__(self, input_size, emb_size, act=lambda x: x, mode='None', layers=2):
        super(FastGAE, self).__init__()
        self.encoder = GraphConvolution(input_size, emb_size, mode=mode, act=act)
        self.medium = nn.ModuleList([GraphConvolution(emb_size, emb_size, mode=mode, act=act) for i in range(layers-2)])
        self.mean = GraphConvolution(emb_size, emb_size, mode='None', act=lambda x: x)

    def forward(self, adj, x=None, device='cuda:0'):
        if x is None:
            x = Variable(torch.rand(adj.shape[0], self.input_size, dtype=torch.float32)).to(device)
        support = self.encoder(x, adj)
        for m in self.medium:
            support = m(support, adj)
        support = self.mean(support, adj)
        return support

