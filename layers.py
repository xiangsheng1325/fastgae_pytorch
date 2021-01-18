import torch, math, copy
import scipy.sparse as sp
import numpy as np
from torch.nn.modules.module import Module
import torch.nn as nn
from torch.nn.parameter import Parameter


def normalize(adj, device='cpu'):
    if isinstance(adj, torch.Tensor):
        adj_ = adj.to(device)
    elif isinstance(adj, sp.csr_matrix):
        adj_ = torch.from_numpy(adj.toarray()).float().to(device)
    elif isinstance(adj, np.ndarray):
        adj_ = torch.from_numpy(adj).float().to(device)
    else:
        adj_ = adj.to(device)
    adj_ = adj_ + torch.eye(adj_.shape[0]).to(device)
    rowsum = adj_.sum(1)
    degree_mat_inv_sqrt = torch.diag(torch.pow(rowsum, -0.5).flatten())
    degree_mat_sqrt = torch.diag(torch.pow(rowsum, -0.5).flatten())
    adj_normalized = torch.mm(torch.spmm(degree_mat_inv_sqrt, adj_), degree_mat_sqrt)
    # return torch.from_numpy(adj_normalized).float().to(device_
    return adj_normalized


def coo_to_csp(sp_coo):
    num = sp_coo.shape[0]
    row = sp_coo.row
    col = sp_coo.col
    sp_tensor = torch.sparse.FloatTensor(torch.LongTensor(np.stack([row, col])),
                                         torch.tensor(sp_coo.data),
                                         torch.Size([num, num]))
    return sp_tensor


#def sp_diag(sp_tensor):
    # sp_tensor = sp_tensor.to_dense()
#    sp_array = sp_tensor.to('cpu').numpy()
#    sp_diags = sp.diags(sp_array).tocoo()
#    return coo_to_csp(sp_diags)


def sp_normalize(adj_def, device='cpu'):
    """
    :param adj: scipy.sparse.coo_matrix
    :param device: default as cpu
    :return: normalized_adj:
    """
    adj_ = sp.coo_matrix(adj_def)
    adj_ = adj_ + sp.coo_matrix(sp.eye(adj_def.shape[0]), dtype=np.float32)
    rowsum = np.array(adj_.sum(axis=1)).reshape(-1)
    norm_unit = np.float_power(rowsum, -0.5).astype(np.float32)
    degree_mat_inv_sqrt = sp.diags(norm_unit)
    degree_mat_sqrt = copy.copy(degree_mat_inv_sqrt)
    # degree_mat_sqrt = degree_mat_inv_sqrt.to_dense()
    support = adj_.__matmul__(degree_mat_sqrt)
    # support = coo_to_csp(support.tocoo())
    # degree_mat_inv_sqrt = coo_to_csp(degree_mat_inv_sqrt.tocoo())
    adj_normalized = degree_mat_inv_sqrt.__matmul__(support)
    adj_normalized = coo_to_csp(adj_normalized.tocoo())
    return adj_normalized, rowsum
    # coo_adj = sp.coo_matrix(adj_normalized.to('cpu').numpy())
    # return coo_to_csp(coo_adj).to(device), rowsum


class PairNorm(nn.Module):
    def __init__(self, mode='PN', scale=1):
        """
            mode:
              'None' : No normalization
              'PN'   : Original version
              'PN-SI'  : Scale-Individually version
              'PN-SCS' : Scale-and-Center-Simultaneously version

            ('SCS'-mode is not in the paper but we found it works well in practice,
              especially for GCN and GAT.)
            PairNorm is typically used after each graph convolution operation.
        """
        assert mode in ['None', 'PN', 'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

        # Scale can be set based on origina data, and also the current feature lengths.
        # We leave the experiments to future. A good pool we used for choosing scale:
        # [0.1, 1, 10, 50, 100]

    def forward(self, x):
        if self.mode == 'None':
            return x

        col_mean = x.mean(dim=0)
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = self.scale * x / rownorm_mean

        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual

        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean

        return x


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, mode='None', act=lambda x: x):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.pn = PairNorm(mode=mode)
        self.act = act
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return self.act(self.pn(output))

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

