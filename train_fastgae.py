from args import get_options
from models import FastGAE
from layers import sp_normalize, coo_to_csp
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import torch.nn.functional as F
import scipy.sparse as sp
import torch.optim as optim
import pickle, os, pprint, torch, datetime, warnings
warnings.filterwarnings("ignore")
import networkx as nx
import numpy as np


opt = get_options()
##{ temporarily change hyper-parameters
opt.DATA = 'google'
graph = pickle.load(open("./data/{}.graph".format(opt.DATA), "rb"))
adj = nx.adjacency_matrix(graph)
adj_float = adj.astype(np.float32)
adj_def = sp.coo_matrix(adj, dtype=np.float32)
opt.av_size = adj.shape[0]
opt.gpu = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
opt.emb_size = 256
opt.lr = 0.01
opt.max_epochs=200000
##}

# optimization
dump = False
pnmode = 'PN-SI'
if opt.gpu == '':
    device = 'cpu'
else:
    device = 'cuda:0'

# fastgae
G = FastGAE(input_size=opt.av_size, emb_size=opt.emb_size, act=F.relu, mode=pnmode).to(device)

opt_gen = optim.Adam(G.parameters(), lr=opt.lr)
scheduler = StepLR(opt_gen, step_size=400, gamma=0.5)
# main

print('=========== OPTIONS ===========')
pprint.pprint(vars(opt))
print(' ======== END OPTIONS ========\n\n')

norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)


# adj = Variable(torch.from_numpy(adj).float())#.to(device)
pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
# early stopping
best_performance = 0.
best_tolerance = 0 # < 1

max_epochs = opt.max_epochs
val_epoch = 800
sample_node_num = opt.sample_size
tolerante = 1
# sp_adj = coo_to_csp(adj_def)
sp_feature = coo_to_csp(sp.coo_matrix(sp.diags(np.ones(adj_def.shape[0])), dtype=np.float32)).to(device)
adj_normalized, degree_vec = sp_normalize(adj_def, device='cpu')
degree_strategy = degree_vec/degree_vec.sum()
adj_normalized = Variable(adj_normalized).to(device)#
for epoch in range(max_epochs):
    starttime = datetime.datetime.now()
    subgraph_nodes = np.random.choice(adj.shape[0],
                                      size=sample_node_num, replace=False,
                                      p=degree_strategy)
    subnode_time = datetime.datetime.now()
    # subgraph_adj = torch.from_numpy(adj[subgraph_nodes, :][:, subgraph_nodes].todense()).to(device)
    subgraph_adj = coo_to_csp(adj_float[subgraph_nodes, :][:, subgraph_nodes].tocoo()).to(device).to_dense()
    subadj_time = datetime.datetime.now()
    subgraph_pos_weight = float(sample_node_num * sample_node_num - subgraph_adj.sum()) / subgraph_adj.sum()
    final_emb = G(adj_normalized, sp_feature, device=device)
    subgraph_emb = final_emb[subgraph_nodes, :]
    train_loss = norm*F.binary_cross_entropy_with_logits(torch.mm(subgraph_emb, subgraph_emb.T), subgraph_adj,
                                                         pos_weight=subgraph_pos_weight)
    loss_time = datetime.datetime.now()
    opt_gen.zero_grad()
    train_loss.backward()
    opt_gen.step()
    scheduler.step()
    # auc, acc = get_scores(adj_def, rec_adj.data.cpu().numpy())
    endtime = datetime.datetime.now()
    if (epoch+1) % val_epoch or train_loss > 0.37:
        print('[%05d/%d]: loss:%.4f, time:%.8s, detailed time:%.4s %.4s'
              % (epoch+1,
                 max_epochs,
                 train_loss,
                 str(endtime-starttime)[-12:], str(endtime-loss_time)[-8:], str(subadj_time-subnode_time)[-8:]))
    else:
        with torch.no_grad():
            final_emb = final_emb.to('cpu')
            # calculating cost
            cost = 0.
        print('[%05d/%d]: loss:%.4f, time:%.8s, cost:%.4f'
              % (epoch + 1,
                 max_epochs,
                 train_loss,
                 str(endtime - starttime)[-12:], cost))
        performance_metric = cost
        # performance_metric = auc
        if performance_metric > best_performance:
            best_performance = performance_metric
            best_tolerance = 0
        else:
            best_tolerance += 1
        if best_tolerance >= tolerante:
            print("*** Early stopping due to no progress...")
            if dump:
                with torch.no_grad():
                    graphs = []
                    for i in range(10):
                        final_emb = G(adj_normalized, sp_feature, device)
                        final_emb = final_emb.to('cpu')
                        rec_adj = torch.mm(final_emb, final_emb.T)
                        # todo: memory saving reconstruction
            break
