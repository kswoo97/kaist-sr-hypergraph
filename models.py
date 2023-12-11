import math
import torch
import copy

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Linear
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter_add, scatter
from torch_geometric.typing import Adj, Size, OptTensor
from typing import Optional, Callable

from sklearn.metrics import average_precision_score as apscore
from sklearn.metrics import roc_auc_score as auroc

from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear as Linear2
from torch_geometric.nn.inits import zeros
from torch_geometric.data import Data

from typing import Optional, Callable

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros


class MeanPoolingConv(MessagePassing):
    _cached_norm_n2e: Optional[Tensor]
    _cached_norm_e2n: Optional[Tensor]

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout: float = 0.0,
                 act: Callable = nn.PReLU(), bias: bool = True, cached: bool = False,
                 row_norm: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        kwargs.setdefault('flow', 'source_to_target')
        super().__init__(node_dim=0, **kwargs)

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.act = act
        self.cached = cached
        self.row_norm = row_norm

        self.lin_n2e = Linear(in_dim, hid_dim, bias=False, weight_initializer='glorot')
        self.lin_e2n = Linear(hid_dim, out_dim, bias=False, weight_initializer='glorot')

        if bias:
            self.bias_n2e = Parameter(torch.Tensor(hid_dim))
            self.bias_e2n = Parameter(torch.Tensor(out_dim))
        else:
            self.register_parameter('bias_n2e', None)
            self.register_parameter('bias_e2n', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_n2e.reset_parameters()
        self.lin_e2n.reset_parameters()
        zeros(self.bias_n2e)
        zeros(self.bias_e2n)
        self._cached_norm_n2e = None
        self._cached_norm_e2n = None

    def forward(self, x: Tensor, hyperedge_index: Tensor,
                num_nodes: Optional[int] = None, num_edges: Optional[int] = None):

        if num_nodes is None:
            num_nodes = x.shape[0]
        if num_edges is None and hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        cache_norm_n2e = self._cached_norm_n2e
        cache_norm_e2n = self._cached_norm_e2n

        if (cache_norm_n2e is None) or (cache_norm_e2n is None):
            hyperedge_weight = x.new_ones(num_edges)

            node_idx, edge_idx = hyperedge_index
            Dn = scatter_add(hyperedge_weight[hyperedge_index[1]],
                             hyperedge_index[0], dim=0, dim_size=num_nodes)
            De = scatter_add(x.new_ones(hyperedge_index.shape[1]),
                             hyperedge_index[1], dim=0, dim_size=num_edges)

            if self.row_norm:
                Dn_inv = 1.0 / Dn
                Dn_inv[Dn_inv == float('inf')] = 0
                De_inv = 1.0 / De
                De_inv[De_inv == float('inf')] = 0

                norm_n2e = De_inv[edge_idx]
                norm_e2n = Dn_inv[node_idx]

            else:
                Dn_inv_sqrt = Dn.pow(-0.5)
                Dn_inv_sqrt[Dn_inv_sqrt == float('inf')] = 0
                De_inv_sqrt = De.pow(-0.5)
                De_inv_sqrt[De_inv_sqrt == float('inf')] = 0

                norm = De_inv_sqrt[edge_idx] * Dn_inv_sqrt[node_idx]
                norm_n2e = norm
                norm_e2n = norm

            if self.cached:
                self._cached_norm_n2e = norm_n2e
                self._cached_norm_e2n = norm_e2n
        else:
            norm_n2e = cache_norm_n2e
            norm_e2n = cache_norm_e2n

        x = self.lin_n2e(x)
        e = self.propagate(hyperedge_index, x=x, norm=norm_n2e,
                           size=(num_nodes, num_edges))  # Node to edge

        if self.bias_n2e is not None:
            e = e + self.bias_n2e
        e = self.act(e)
        e = F.dropout(e, p=self.dropout, training=self.training)

        x = self.lin_e2n(e)
        n = self.propagate(hyperedge_index.flip([0]), x=x, norm=norm_e2n,
                           size=(num_edges, num_nodes))  # Edge to node

        if self.bias_e2n is not None:
            n = n + self.bias_e2n

        return n, e  # No act, act

    def message(self, x_j: Tensor, norm: Tensor):
        return norm.view(-1, 1) * x_j

class HyperEncoder(nn.Module):
    def __init__(self, in_dim, edge_dim, node_dim, num_layers=2, act: Callable = nn.PReLU()):
        super(HyperEncoder, self).__init__()
        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.num_layers = num_layers
        self.act = act

        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(MeanPoolingConv(self.in_dim, self.edge_dim, self.node_dim, cached=False, act=act))
        else:
            self.convs.append(MeanPoolingConv(self.in_dim, self.edge_dim, self.node_dim, cached=False, act=act))
            for _ in range(self.num_layers - 2):
                self.convs.append(MeanPoolingConv(self.node_dim, self.edge_dim, self.node_dim, cached=False, act=act))
            self.convs.append(MeanPoolingConv(self.node_dim, self.edge_dim, self.node_dim, cached=False, act=act))
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, hyperedge_index: Tensor, num_nodes: int, num_edges: int):
        for i in range(self.num_layers):
            x, e = self.convs[i](x, hyperedge_index, num_nodes, num_edges)
            x = self.act(x)
        return x, e  # act, act
    
class HyperDecoder(nn.Module):
    def __init__(self, in_dim, edge_dim, node_dim, num_layers=2, act: Callable = nn.PReLU()):
        super(HyperDecoder, self).__init__()
        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.num_layers = num_layers
        self.act = act
        self.Smoother = torch.nn.LSTM(input_size = in_dim, hidden_size = in_dim, num_layers=1, batch_first = False)
        
        self.input_mask = torch.nn.Parameter(torch.zeros(node_dim), requires_grad = True)
        self.embedding_mask = torch.nn.Parameter(torch.zeros(in_dim), requires_grad = True)

        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(MeanPoolingConv(self.in_dim, self.edge_dim, self.node_dim, cached=False, act=act))
        else:
            self.convs.append(MeanPoolingConv(self.in_dim, self.in_dim, self.in_dim, cached=False, act=act))
            for _ in range(self.num_layers - 2):
                self.convs.append(MeanPoolingConv(self.in_dim, self.in_dim, self.in_dim, cached=False, act=act))
            self.convs.append(MeanPoolingConv(self.in_dim, self.edge_dim, self.node_dim, cached=False, act=act))
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, hyperedge_index: Tensor, num_nodes: int, num_edges: int):
        for i in range(self.num_layers):
            x, e = self.convs[i](x, hyperedge_index, num_nodes, num_edges)
            x = self.act(x)
        return x, e  # act, act
    
class TaskMLP(nn.Module) : 
    
    def __init__(self, in_dim, hidden_dim, n_layer, drop_p = 0.5) :
        super(TaskMLP, self).__init__()
        
        self.linears = nn.ModuleList()
        for i in range(n_layer) : 
            if i == 0 : 
                self.linears.append(torch.nn.Linear(in_dim, hidden_dim))
            else : 
                self.linears.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.act = torch.nn.ReLU()
        self.num_layers = n_layer
        self.DropLayer = torch.nn.Dropout(p=drop_p)
        
    def forward(self, x, ind1, ind2) : 
        
        for i in range(self.num_layers):
            
            if i < self.num_layers - 1 :
                x = self.linears[i](x)
                x = self.DropLayer(x)
                x = self.act(x)
            else : 
                x = self.linears[i](x)
        pred = torch.sum(x[ind1, :] * x[ind2, :], 1)
        return torch.sigmoid(pred)
    
class TaskLSTM(nn.Module) : 
    
    def __init__(self, in_dim, hidden_dim, drop_p = 0.5) :
        super(TaskLSTM, self).__init__()
        
        self.linears = nn.ModuleList()
        
        self.DropLayer = torch.nn.Dropout(p=drop_p)
        self.LSTM = torch.nn.LSTM(input_size = in_dim, 
                                        hidden_size = hidden_dim, 
                                       num_layers = 1)
        
    def forward(self, x, ind1, ind2) : 
        
        x = self.LSTM(x)[1][0][0]
        x = self.DropLayer(x)
        pred = torch.sum(x[ind1, :] * x[ind2, :], 1)
        return torch.sigmoid(pred)
    
def createZ (each_X, each_HE, last_time, encoder, device, feature_type = 'MLP', model_type = 'gd',time_mapper = None) : 
    
    each_Z = []
    
    if model_type == 'gd' : 
    
        with torch.no_grad() : 
            encoder.eval()

            for x_feat, e_edge in zip(each_X[:last_time + 1], each_HE[:last_time + 1]) : 
                x_ = x_feat.to(device)
                e_ = e_edge.to(device)
                n_node, n_edge = x_.shape[0], torch.max(e_[1]) + 1
                Z = encoder(x_, e_, n_node, n_edge)[0].to('cpu')
                each_Z.append(Z)
                del x_, e_, Z
    
    else : # Without self-supervision.
        
        for x_feat, e_edge in zip(each_X[:last_time + 1], each_HE[:last_time + 1]) : 
            each_Z.append(x_feat)
            
    if feature_type == 'MLP' : 
        
        return each_Z
    
    elif feature_type == 'LSTM' : 
        
        LSTM_Z = []

        for target_time in (range(len(each_Z))) : 

            if target_time == 0 : 
                TotalX = each_Z[0].unsqueeze(0)

            else : 
                TotalX = torch.zeros((2, each_Z[1].shape[0], each_Z[1].shape[1]))
                T_prev = time_mapper[target_time][0]
                T_new = time_mapper[target_time][1]
                TotalX[1, :, :] = each_Z[1]
                TotalX[0, T_new, :] = each_Z[0][T_prev]
                each_Z.pop(0)

            LSTM_Z.append(TotalX)

        return LSTM_Z
    
    else : 
        raise TypeError("Feature type should be given enither MLP or LSTM")