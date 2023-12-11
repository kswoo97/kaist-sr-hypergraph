import os
import copy
import time
import torch
import pickle
import random
import numpy as np

from torch import Tensor
from tqdm import tqdm
from torch_geometric.typing import OptTensor
from torch_scatter import scatter_add
from itertools import permutations

## Related Tools

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def drop_features(x: Tensor, p: float):
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < p
    x = x.clone()
    x[:, drop_mask] = 0
    return x


def filter_incidence(row: Tensor, col: Tensor, hyperedge_attr: OptTensor, mask: Tensor):
    return row[mask], col[mask], None if hyperedge_attr is None else hyperedge_attr[mask]


def drop_incidence(hyperedge_index: Tensor, p: float = 0.2):
    if p == 0.0:
        return hyperedge_index

    row, col = hyperedge_index
    mask = torch.rand(row.size(0), device=hyperedge_index.device) >= p

    row, col, _ = filter_incidence(row, col, None, mask)
    hyperedge_index = torch.stack([row, col], dim=0)
    return hyperedge_index


def drop_nodes(hyperedge_index: Tensor, num_nodes: int, num_edges: int, p: float):
    if p == 0.0:
        return hyperedge_index

    drop_mask = torch.rand(num_nodes, device=hyperedge_index.device) < p
    drop_idx = drop_mask.nonzero(as_tuple=True)[0]

    H = torch.sparse_coo_tensor(hyperedge_index, \
                                hyperedge_index.new_ones((hyperedge_index.shape[1],)),
                                (num_nodes, num_edges)).to_dense()
    H[drop_idx, :] = 0
    hyperedge_index = H.to_sparse().indices()

    return hyperedge_index


def drop_hyperedges(hyperedge_index: Tensor, num_nodes: int, num_edges: int, p: float):
    if p == 0.0:
        return hyperedge_index

    drop_mask = torch.rand(num_edges, device=hyperedge_index.device) < p
    drop_idx = drop_mask.nonzero(as_tuple=True)[0]

    H = torch.sparse_coo_tensor(hyperedge_index, \
                                hyperedge_index.new_ones((hyperedge_index.shape[1],)),
                                (num_nodes, num_edges)).to_dense()
    H[:, drop_idx] = 0
    hyperedge_index = H.to_sparse().indices()

    return hyperedge_index


def valid_node_edge_mask(hyperedge_index: Tensor, num_nodes: int, num_edges: int):
    ones = hyperedge_index.new_ones(hyperedge_index.shape[1])
    Dn = scatter_add(ones, hyperedge_index[0], dim=0, dim_size=num_nodes)
    De = scatter_add(ones, hyperedge_index[1], dim=0, dim_size=num_edges)
    node_mask = Dn != 0
    edge_mask = De != 0
    return node_mask, edge_mask


def hyperedge_index_masking(hyperedge_index, num_nodes, num_edges, node_mask, edge_mask):
    if node_mask is None and edge_mask is None:
        return hyperedge_index

    H = torch.sparse_coo_tensor(hyperedge_index, \
                                hyperedge_index.new_ones((hyperedge_index.shape[1],)),
                                (num_nodes, num_edges)).to_dense()
    if node_mask is not None and edge_mask is not None:
        masked_hyperedge_index = H[node_mask][:, edge_mask].to_sparse().indices()
    elif node_mask is None and edge_mask is not None:
        masked_hyperedge_index = H[:, edge_mask].to_sparse().indices()
    elif node_mask is not None and edge_mask is None:
        masked_hyperedge_index = H[node_mask].to_sparse().indices()
    return masked_hyperedge_index

### HyperMAE Trainer

class HyperMAETrainer(object):

    def __init__(self, Xs, Es, IDXs, encoder, decoder, device, dim = 128, do_e_aug = False):  ## Data is given as torch scatter data class type

        self.Xs = Xs
        self.Es = Es
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.idxs = [{v : i for i, v in enumerate(IDX)} for IDX in IDXs]
        self.node_idxs = [list(np.arange(V.shape[0])) for V in Xs]
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        self.dim = dim
        self.do_e_aug = do_e_aug

    def do_forward_pass(self, time_limit):

        torch.autograd.set_detect_anomaly(True)
        loss_lists = []
        
        idx = 0
            
        for x, e in zip(self.Xs[:time_limit + 1], self.Es[:time_limit + 1]) : 

            self.optimizer.zero_grad()
            #curX, curE = x.to(self.device) , e.to(self.device)
            if self.do_e_aug : 
                curX, curE = torch.clone(x).to(self.device), drop_incidence(e, 0.3).to(self.device)
            else : 
                curX, curE = torch.clone(x).to(self.device), e.to(self.device)
            masked_idx = list(np.random.choice(a = curX.shape[0], size = int(self.drop_p * curX.shape[0]), replace = False))
            masked_idx.sort()

            num_nodes = curX.shape[0]
            num_edges = torch.max(curE[1]) + 1
            
            cur_mask1 = torch.ones(num_nodes).to(self.device)
            cur_mask2 = torch.ones(num_nodes).to(self.device)
            cur_mask1[masked_idx] = 0.0
            cur_mask2[masked_idx] = 0.0
            n1, _ = self.encoder(curX * (cur_mask1.unsqueeze(-1)), curE, num_nodes, num_edges)
            n2, _ = self.decoder(n1 * (cur_mask2.unsqueeze(-1)), curE, num_nodes, num_edges)
            
            totalL = torch.mean((1 - self.cos(curX[masked_idx, :], n2[masked_idx, :])) ** self.gamma)
            totalL.backward()
            self.optimizer.step()
            
            loss_lists.append(totalL.to('cpu').item())

            del curX, curE, n1, n2, _
            
            idx += 1
        
        return np.sum(loss_lists)

    def fit(self, epoch, lr, drop_feature=0.25, gamma=1, save_model=False, explicit_bound = None) :

        loss_lists = []
        self.drop_p = drop_feature
        self.gamma = gamma
        
        params = copy.deepcopy(self.encoder.state_dict())            
        sub_params = copy.deepcopy(self.decoder.state_dict())            
        self.parameters = [[] for _ in range(len(explicit_bound))]

        for i, time_limit in enumerate(explicit_bound) : ## Training for each batch

            self.encoder.load_state_dict(params)
            self.decoder.load_state_dict(sub_params)
            self.encoder.train()        
            self.decoder.train()  
            self.optimizer = torch.optim.AdamW(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr = lr, weight_decay = 1e-6)

            for ep in tqdm(range(epoch)):

                l1 = self.do_forward_pass(time_limit)
                loss_lists.append(l1)

            param = copy.deepcopy(self.encoder.state_dict())
            self.parameters[i].append(param)

        return self.parameters, loss_lists