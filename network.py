import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from pool import GraphMultisetTransformer
from torch_geometric.utils import to_dense_batch, to_scipy_sparse_matrix
from torch_geometric.nn import global_max_pool as gmp
import scipy.sparse as sp
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.init as init
import math

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim, bias=True), nn.ReLU(inplace=True),
                                 nn.Linear(hidden_dim, output_dim, bias=True), nn.ReLU(inplace=True))
        
    def forward(self, x):
        return self.mlp(x)


class GraphCNN(nn.Module):
    def __init__(self, channel_dims=[], fc_dim=512, num_classes=256, num_supernode=4096, device_model=None, evratio=1.0, args=None):
        super(GraphCNN, self).__init__()

        self.device = device_model
        self.evratio = evratio
        self.pool = GraphMultisetTransformer(512, 256, 512, 1024, None, num_supernode, ['GMPool_evec', 'GMPool_I'], num_heads=args.num_head, layer_norm=True, args=args)
        self.drop1 = nn.Dropout(p=0.2)
        self.args = args
        self.temperature = args.temperature
    

    def forward(self, x, data, num_vertices, num_edges, evectors):
        batch_number = data.batch[-1] + 1
        new_element = torch.tensor([0], device=self.device)
        num_edges = torch.cat((new_element, num_edges))
        num_vertices = torch.cat((new_element, num_vertices))
        
        feat_dim = x.shape[-1]
        
        x = self.drop1(x)
        
        transformed_evectors = []
        transformed_graphs = []
        base_x = []
        for i in range(len(num_vertices) - 1):
            start, end = num_vertices[i], num_vertices[i + 1]

            evec = torch.tensor(evectors[i], device=self.device).type(torch.float32)
            N, M = evec.shape
            num_columns = self.args.num_node
            step = (M - 1) / (num_columns - 1)
            selected_indices = [round(i * step) for i in range(num_columns)]
           
            evec = evec[:, selected_indices]

            evec = torch.nn.functional.pad(
                evec, (0, 0, 0, 1024 - N), mode="constant", value=0
            )
            evec = evec.transpose(0, 1)
            transformed_evectors.append(evec)

            graph_x = x[start:end]
            result_x = torch.cat([graph_x, graph_x, graph_x], dim=0)
            base_x.append(result_x)
            Ni = graph_x.shape[0]
            
            evec = torch.tensor(evectors[i], device=self.device).type(torch.float32)
            
            gft_graph = torch.matmul(evec.T, graph_x)
            dim_split = gft_graph.shape[0]
            first_ten = math.ceil(dim_split * self.evratio)
            second_twenty = math.ceil(dim_split * self.evratio * 2)

            split_sizes = [first_ten, second_twenty, dim_split - first_ten - second_twenty]
            
            gft_splits = torch.split(gft_graph, split_sizes, dim=0)
            evec_splits = torch.split(evec, split_sizes, dim=1)
            
            updated_parts = [torch.matmul(evec_split, gft_split) for evec_split, gft_split in zip(evec_splits, gft_splits)]
            updated_graph = torch.cat(updated_parts, dim=0)
            
            transformed_graphs.append(updated_graph)

        x_updated = torch.cat(transformed_graphs, dim=0)
        original_x = torch.cat(base_x, dim=0)
        updated_evectors = torch.stack(transformed_evectors)

        
        g_level_feat_bp, g_level_feat_mf, g_level_feat_cc, decorrelation_loss = self.pool(x_updated, original_x, updated_evectors, data.batch, data.edge_index.long(), num_vertices, self.temperature)
        
        n_level_feat = x


        return n_level_feat, g_level_feat_bp, g_level_feat_mf, g_level_feat_cc, decorrelation_loss


class CL_protNET(torch.nn.Module):
    def __init__(self, out_dim, esm_embed=True, num_supernode=1024, device_model=None, args=None):
        super(CL_protNET,self).__init__()
        self.esm_embed = esm_embed
        self.out_dim = out_dim
        self.one_hot_embed = nn.Embedding(21, 96)
        self.proj_aa = nn.Linear(96, 512) 
        self.device = device_model

        if esm_embed:
            self.proj_esm = nn.Linear(1280, 512)
            self.gcn = GraphCNN(num_supernode=num_supernode, device_model=device_model, evratio=args.eigenvec_ratio, args=args)
        else:
            self.gcn = GraphCNN(num_supernode=num_supernode, device_model=device_model, evratio=args.eigenvec_ratio, args=args)

        self.readout_bp = nn.Sequential(
                        nn.Linear(512, 1024),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(1024, 1943),
                        nn.Sigmoid()
        )
        self.readout_mf = nn.Sequential(
                        nn.Linear(512, 1024),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(1024, 489),
                        nn.Sigmoid()
        )
        self.readout_cc = nn.Sequential(
                        nn.Linear(512, 1024),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(1024, 320),
                        nn.Sigmoid()
        )
        self.softmax = nn.Softmax(dim=-1)
     
    def forward(self, data, num_vertices, num_edges, lp_evecs):
        #x_aa = self.one_hot_embed(data.native_x.long())
        #x_aa = self.proj_aa(x_aa)
        
        if self.esm_embed:
            x = data.x.float()
            x_esm = self.proj_esm(x)
        #    x = F.relu(x_aa + x_esm)
            x = F.relu(x_esm)
            
        else:
            x = F.relu(x_aa)
    
        gcn_n_feat1, gcn_g_feat_bp1, gcn_g_feat_mf1, gcn_g_feat_cc1, decorrelation_loss = self.gcn(x, data, num_vertices, num_edges, lp_evecs)
        
       
        y_pred_bp = self.readout_bp(gcn_g_feat_bp1)
        y_pred_mf = self.readout_mf(gcn_g_feat_mf1)
        y_pred_cc = self.readout_cc(gcn_g_feat_cc1)

        return y_pred_bp, y_pred_mf, y_pred_cc, decorrelation_loss
