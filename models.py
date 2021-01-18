from layers import GraphConvolution, PairNorm
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch


class GCNEncoder(nn.Module):
    def __init__(self, input_size, emb_size, pool_size, mode='PN-SI'):
        super(GCNEncoder, self).__init__()
        self.input_size = input_size
        # self.encode00 = nn.Linear(input_size, emb_size)
        # self.encode01 = GraphConvolution(emb_size, emb_size)
        self.encode1 = GraphConvolution(input_size, emb_size)
        self.pn = PairNorm(mode=mode)
        self.pool1 = GraphConvolution(emb_size, pool_size)
        self.encode2 = GraphConvolution(emb_size, pool_size)
        for m in self.modules():
            if isinstance(m, GraphConvolution):
                m.reset_parameters()

    def forward(self, adj, x=None, device='cuda:0'):
        # x = Variable(torch.diag(torch.ones(adj.shape[0]).float()).float()).to(device)
        if x is None:
            x = Variable(torch.rand(adj.shape[0], self.input_size, dtype=torch.float32)).to(device)
        # adj = normalize(adj, device)
        # enc1 = self.encode01(self.encode00(x), adj)
        enc1 = self.encode1(x, adj)
        enc1 = self.pn(enc1)
        enc1 = F.relu(enc1, inplace=True)
        pool1 = self.pool1(enc1, adj)
        pool1 = self.pn(pool1)
        pool1 = F.softmax(pool1)
        enc2 = self.encode2(enc1, adj)
        enc2 = self.pn(enc2)
        enc2 = F.relu(enc2, inplace=True)
        # output = torch.mul(pool1, enc2)
        return enc2, pool1, adj


class GCNDecoder(nn.Module):
    def __init__(self, emb_size, pool_size, mode='PN-SI'):
        super(GCNDecoder, self).__init__()
        self.decode1 = GraphConvolution(pool_size, emb_size)
        self.decode2 = GraphConvolution(emb_size, emb_size)
        self.pn = PairNorm(mode=mode)
        for m in self.modules():
            if isinstance(m, GraphConvolution):
                m.reset_parameters()

    def forward(self, enc, pool, adj):
        emb = torch.mul(enc, pool)
        emb = F.relu(self.pn(self.decode1(emb, adj)), inplace=True)
        emb = self.decode2(emb, adj)
        return emb
        # score = torch.mm(emb, emb.T)
        # return F.sigmoid(score)


class GCNGenerator(nn.Module):
    def __init__(self, input_size, emb_size, pool_size, mode='PN-SI'):
        super(GCNGenerator, self).__init__()
        self.encoder = GCNEncoder(input_size, emb_size, pool_size, mode)
        self.decoder = GCNDecoder(emb_size, pool_size, mode)

    def forward(self, adj, x=None, device='cuda:0'):
        return self.decoder(*(self.encoder(adj, x, device)))


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


class HieGAE(nn.Module):
    def __init__(self, input_size, emb_size, pool_size, mode='PN-SI'):
        super(HieGAE, self).__init__()
        self.encoder = GCNEncoder(input_size, emb_size, pool_size, mode)
        self.decoder = GraphConvolution(pool_size, emb_size)

    def forward(self, adj, x=None, device='cuda:0'):
        enc, pool, adj = self.encoder(adj, x, device)
        support = torch.mul(enc, pool)
        return self.decoder(support, adj)
