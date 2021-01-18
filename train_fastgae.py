from args import get_options
from models import FastGAE
from layers import sp_normalize, coo_to_csp
from input_output import topk_adj, community_detect, get_scores
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import torch.nn.functional as F
import scipy.sparse as sp
import torch.optim as optim
import pickle, community, os, pprint, torch, datetime, warnings
warnings.filterwarnings("ignore")
import networkx as nx
import numpy as np


opt = get_options()
##{ 临时改超参数
opt.DATA = 'google'
graph = pickle.load(open("/home/xiangsheng/venv/ggen/ggen/generators/data/{}.graph".format(opt.DATA), "rb"))
adj = nx.adjacency_matrix(graph)
adj_float = adj.astype(np.float32)
adj_def = sp.coo_matrix(adj, dtype=np.float32)
ref_labels = None
opt.av_size = adj.shape[0]
opt.gpu = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
# commu_size = len(set(community.generate_dendrogram(graph)[0].values()))
# opt.pool_size = min(max(2**(int(np.log2(commu_size))+1), 128), 2048)
opt.emb_size = 256
opt.lr = 0.01
opt.max_epochs=200000
## 正式训练时收起 }

# optimization
dump = False
pnmode = 'PN-SI'
if opt.gpu == '':
    device = 'cpu'
else:
    device = 'cuda:0'

# G = GCNGenerator(input_size=opt.av_size, emb_size=opt.emb_size, pool_size=opt.pool_size, mode=pnmode).to(device)
# fastgae
G = FastGAE(input_size=opt.av_size, emb_size=opt.emb_size, act=F.relu, mode=pnmode).to(device)
# 1429 MiB
opt_gen = optim.Adam(G.parameters(), lr=opt.lr)
scheduler = StepLR(opt_gen, step_size=10000, gamma=0.5)
# main

print('=========== OPTIONS ===========')
pprint.pprint(vars(opt))
print(' ======== END OPTIONS ========\n\n')

norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)


# adj = Variable(torch.from_numpy(adj).float())#.to(device)# 2913 MiB
# pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
# weight_mask = adj == 1
# weight_tensor = torch.ones(weight_mask.shape).to(device)# 4769 MiB
# weight_tensor[weight_mask] = pos_weight# 4789 MiB
# weight_tensor = weight_tensor.to(device)
# early stopping
best_performance = 0.
best_tolerance = 0 # < 10

max_epochs = opt.max_epochs
val_epoch = 10000
sample_node_num = 16384
tolerante = 1
# sp_adj = coo_to_csp(adj_def)
sp_feature = coo_to_csp(sp.coo_matrix(sp.diags(np.ones(adj_def.shape[0])), dtype=np.float32)).to(device)
adj_normalized, degree_vec = sp_normalize(adj_def, device='cpu')
degree_strategy = degree_vec/degree_vec.sum()
adj_normalized = Variable(adj_normalized).to(device)# 10731 MiB
for epoch in range(max_epochs):
    starttime = datetime.datetime.now()
    subgraph_nodes = np.random.choice(adj.shape[0],
                                      size=sample_node_num, replace=False,
                                      p=degree_strategy)
    # subgraph_nodes = torch.from_numpy(subgraph_nodes)
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
        if ref_labels is None:
            ref_labels = community_detect(graph)
        final_emb = final_emb.to('cpu')
        rec_adj = torch.mm(final_emb, final_emb.T)
        c_adj = topk_adj(rec_adj, k=adj.sum())
        pred_labels = community_detect(nx.from_numpy_array(c_adj))
        nmi = normalized_mutual_info_score(pred_labels, ref_labels)
        ami = adjusted_mutual_info_score(pred_labels, ref_labels)
        print('[%05d/%d]: loss:%.4f, time:%.8s, nmi:%.4f, ami:%.4f'
              % (epoch + 1,
                 max_epochs,
                 train_loss,
                 str(endtime - starttime)[-12:],
                 nmi,
                 ami))
        performance_metric = nmi+ami
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
                        c_adj = topk_adj(rec_adj, k=adj.sum().data.cpu())
                        gen_graph = nx.from_numpy_array(c_adj)
                        graphs.append(gen_graph)
                # general experiment
                #pickle.dump(
                #    graphs,
                #    open("/home/xiangsheng/venv/ggen/ggen/generators/result/{}_to_hiegan.graphs".format(opt.DATA), "wb")
                #)
                # tuning experiment
                #pickle.dump(
                #    graphs,
                #    open("/home/xiangsheng/venv/ggen/ggen/generators/result/tuning/hiegan-emb/{}_to_hiegan_emb{}.graphs".format(opt.DATA_DIR, opt.pool_size),
                #         "wb")
                #)
                #pickle.dump(
                #    graphs,
                #    open("/home/xiangsheng/venv/ggen/ggen/generators/result/tuning/hiegan-lr/{}_to_hiegan_lr{}_epoch{}.graphs".format(opt.DATA_DIR, opt.lr, epoch+1),
                #         "wb")
                #)
            break
