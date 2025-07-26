import math
from typing import List, Optional, Tuple, Type

import torch
from torch import Tensor
from torch.nn import LayerNorm, Linear
import torch.nn.init as init

from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch
import torch.nn.functional as F

def weight_decorrelation_loss(W, a, b, c):
    # W의 row 벡터 간 상관관계 감소 (그룹별)
    W_norm = F.normalize(W, dim=1)  # Row-wise 정규화
    num_rows = W.size(0)  # 전체 row 개수
    
    # 그룹의 시작 및 끝 인덱스 설정
    group1 = W[:, :a, :]        # (64, a, D)
    group2 = W[:, a:a+b, :]     # (64, b, D)
    group3 = W[:, a+b:a+b+c, :] # (64, c, D)
    
    decorrelation_loss = 0.0
    
    for group in [group1, group2, group3]:
        if group.size(1) > 1:
            corr_matrix = torch.einsum("ijk,ilk->ijl", group, group)
            identity = torch.eye(group.size(1), device=group.device).unsqueeze(0)
            decorrelation_loss += torch.norm(corr_matrix - identity, dim=(1, 2)).mean()
    return decorrelation_loss

class MAB(torch.nn.Module):
    r"""Multihead-Attention Block."""
    def __init__(self, dim_Q: int, dim_K: int, dim_V: int, num_heads: int,
                 Conv: Optional[Type] = None, layer_norm: bool = False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.layer_norm = layer_norm

        self.fc_q = Linear(dim_Q, dim_V)

        if Conv is None:
            self.layer_k = Linear(dim_K, dim_V)
            self.layer_v = Linear(dim_K, dim_V)
        else:
            self.layer_k = Conv(dim_K, dim_V)
            self.layer_v = Conv(dim_K, dim_V)

        if layer_norm:
            self.ln0 = LayerNorm(dim_V)
            self.ln1 = LayerNorm(dim_V)

        self.fc_o = Linear(dim_V, dim_V)

    def reset_parameters(self):
        self.fc_q.reset_parameters()
        init.zeros_(self.fc_q.bias)
        self.layer_k.reset_parameters()
        init.zeros_(self.layer_k.bias)
        self.layer_v.reset_parameters()
        init.zeros_(self.layer_v.bias)
        if self.layer_norm:
            self.ln0.reset_parameters()
            init.zeros_(self.ln0.bias)
            self.ln1.reset_parameters()
            init.zeros_(self.ln1.bias)
        self.fc_o.reset_parameters()
        init.zeros_(self.fc_o.bias)
        pass

    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        graph: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
        mask: Optional[Tensor] = None,
        temperature: float = 0.1,
    ) -> Tensor:

        Q = self.fc_q(Q)

        if graph is not None:
            x, edge_index, batch = graph
            K, V = self.layer_k(x, edge_index), self.layer_v(x, edge_index)
            K, _ = to_dense_batch(K, batch)
            V, _ = to_dense_batch(V, batch)
        else:
            K, V = self.layer_k(K), self.layer_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), dim=0)
        K_ = torch.cat(K.split(dim_split, 2), dim=0)
        V_ = torch.cat(V.split(dim_split, 2), dim=0)

        
        A = torch.softmax(
            Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), -1)

        out = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)

        if self.layer_norm:
            out = self.ln0(out)

        out = out + self.fc_o(out).relu()

        if self.layer_norm:
            out = self.ln1(out)

        return out

class MAB_graph(torch.nn.Module):
    r"""Multihead-Attention Block."""
    def __init__(self, dim_Q: int, dim_K: int, dim_V: int, dim_evec: int, num_heads: int,
                 Conv: Optional[Type] = None, layer_norm: bool = False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.layer_norm = layer_norm
        
        self.fc_q_G = Linear(dim_evec, dim_V, bias=False)
        self.fc_q_D = Linear(dim_evec, dim_V, bias=False)
        self.fc_q_L = Linear(dim_evec, dim_V, bias=False)

        if Conv is None:
            self.layer_k_G = Linear(dim_K, dim_V, bias=False)
            self.layer_k_D = Linear(dim_K, dim_V, bias=False)
            self.layer_k_L = Linear(dim_K, dim_V, bias=False)
            self.layer_v_G = Linear(dim_K, dim_V)
            self.layer_v_D = Linear(dim_K, dim_V)
            self.layer_v_L = Linear(dim_K, dim_V)
        else:
            self.layer_k = Conv(dim_K, dim_V, bias=False)
            self.layer_v = Conv(dim_K, dim_V)

        if layer_norm:
            self.ln0 = LayerNorm(dim_V)
            self.ln1 = LayerNorm(dim_V)

        self.fc_o = Linear(dim_V, dim_V)

    def reset_parameters(self):
        self.fc_q_G.reset_parameters()
        self.fc_q_D.reset_parameters()
        self.fc_q_L.reset_parameters()
        self.layer_k_G.reset_parameters()
        self.layer_k_D.reset_parameters()
        self.layer_k_L.reset_parameters()
        
        self.layer_v_G.reset_parameters()
        init.zeros_(self.layer_v_G.bias)
        self.layer_v_D.reset_parameters()
        init.zeros_(self.layer_v_G.bias)
        self.layer_v_L.reset_parameters()
        init.zeros_(self.layer_v_G.bias)
        if self.layer_norm:
            self.ln0.reset_parameters()
            init.zeros_(self.ln0.bias)
            self.ln1.reset_parameters()
            init.zeros_(self.ln1.bias)
        self.fc_o.reset_parameters()
        init.zeros_(self.fc_o.bias)
        pass

    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        origin_x: Tensor,
        graph: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
        mask: Optional[Tensor] = None,
        mask_cross: Optional[Tensor] = None, 
        numv: Optional[Tensor] = None,
        a: int = 3,
        b: int = 6,
        c: int = 12,
        temperature: float = 0.1,
    ) -> Tensor:

        query = []
        query.append(self.fc_q_G(Q[:,:a]))
        query.append(self.fc_q_D(Q[:,a:a+b]))
        query.append(self.fc_q_L(Q[:,a+b:a+b+c]))
        Q = torch.cat(query, dim=1)
        differences = [numv[i] - numv[i - 1] for i in range(1, len(numv))]
        
        output_key = torch.zeros_like(K)
        output_value = torch.zeros_like(origin_x)

            # Precompute index ranges for slicing
        key_slices = []
        value_slices = []
        for diff in differences:
            key_slices.append((0, diff, 2 * diff, 3 * diff))
            value_slices.append((0, diff, 2 * diff, 3 * diff))

        for i, (key_slice, value_slice) in enumerate(zip(key_slices, value_slices)):
            k0, k1, k2, k3 = key_slice
            v0, v1, v2, v3 = value_slice

            key_parts = [
                self.layer_k_G(K[i, k0:k1]),
                self.layer_k_D(K[i, k1:k2]),
                self.layer_k_L(K[i, k2:k3])
            ]
            output_key[i, k0:k3] = torch.cat(key_parts, dim=0)
            
            value_parts = [
                self.layer_v_G(origin_x[i, v0:v1]),
                self.layer_v_D(origin_x[i, v1:v2]),
                self.layer_v_L(origin_x[i, v2:v3])
            ]
            output_value[i, v0:v3] = torch.cat(value_parts, dim=0)

        dim_split = self.dim_V // self.num_heads

        Q_ = Q.split(dim_split, 2)
        weight_decorrelation = 0
        for i in range(self.num_heads):
            weight_decorrelation += weight_decorrelation_loss(Q_[i], a, b, c)
        Q_ = torch.cat(Q_, dim=0)
        K_ = torch.cat(output_key.split(dim_split, 2), dim=0)
        V_ = torch.cat(output_value.split(dim_split, 2), dim=0)


        #mask = torch.cat([mask for _ in range(self.num_heads)], 0)
        attention_score = Q_.bmm(K_.transpose(1, 2))
        attention_score = attention_score / math.sqrt(self.dim_V)
        attention_score = attention_score / temperature
        mask_cross = torch.cat([mask_cross for _ in range(self.num_heads)], 0)
        A = torch.softmax(attention_score + mask_cross,-1)

        out = torch.cat((A.bmm(V_)).split(Q.size(0), 0), 2)

        if self.layer_norm:
            out = self.ln0(out)

        out = out + self.fc_o(out).relu()
        out = F.normalize(out, p=2, dim=2)

        if self.layer_norm:
            out = self.ln1(out)

        
        return out, weight_decorrelation

class SAB(torch.nn.Module):
    r"""Self-Attention Block."""
    def __init__(self, in_channels: int, out_channels: int, num_heads: int,
                 Conv: Optional[Type] = None, layer_norm: bool = False):
        super().__init__()
        self.mab = MAB(in_channels, in_channels, out_channels, num_heads,
                       Conv=Conv, layer_norm=layer_norm)

    def reset_parameters(self):
        self.mab.reset_parameters()

    def forward(
        self,
        x: Tensor,
        graph: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
        mask: Optional[Tensor] = None,
        temperature: float = 0.1,
    ) -> Tensor:
        return self.mab(x, x, graph, mask, temperature)

class PMAGroup(torch.nn.Module):
    def __init__(self, channels: int, num_heads: int, num_seeds: int,
                 Conv: Optional[Type] = None, layer_norm: bool = False):
        super(PMAGroup, self).__init__()
        self.pma1 = PMA(channels, num_heads, num_seeds, Conv=None, layer_norm=layer_norm)
        self.pma2 = PMA(channels, num_heads, num_seeds, Conv=None, layer_norm=layer_norm)
        self.pma3 = PMA(channels, num_heads, num_seeds, Conv=None, layer_norm=layer_norm)

    def forward(self, x, graph, mask, temperature):
        x_bp = x.clone()
        x_mf = x.clone()
        x_cc = x.clone()
        x1 = self.pma1(x_bp, graph, mask, temperature)
        x2 = self.pma2(x_mf, graph, mask, temperature)
        x3 = self.pma3(x_cc, graph, mask, temperature)
        return x1, x2, x3

class PMA(torch.nn.Module):
    r"""Graph pooling with Multihead-Attention."""
    def __init__(self, channels: int, num_heads: int, num_seeds: int,
                 Conv: Optional[Type] = None, layer_norm: bool = False):
        super().__init__()
        self.S = torch.nn.Parameter(torch.Tensor(1, num_seeds, channels))
        self.mab = MAB(channels, channels, channels, num_heads, Conv=Conv,
                       layer_norm=layer_norm)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.S)
        self.mab.reset_parameters()

    def forward(
        self,
        x: Tensor,
        graph: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
        mask: Optional[Tensor] = None,
        temperature: float = 0.1,
    ) -> Tensor:
        return self.mab(self.S.repeat(x.size(0), 1, 1), x, graph, mask, temperature)

class GraphMA(torch.nn.Module):
    r"""Graph pooling with Masked Multihead-Attention."""
    def __init__(self, channels: int, evec_channels: int, num_heads: int,
                 Conv: Optional[Type] = None, layer_norm: bool = False):
        super().__init__()
        self.mab = MAB_graph(channels, channels, channels, evec_channels, num_heads, Conv=None,
                       layer_norm=layer_norm)

        self.reset_parameters()

    def reset_parameters(self):
        self.mab.reset_parameters()

    def forward(
        self,
        x: Tensor,
        origin_x: Tensor,
        evectors: Tensor,
        graph: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
        mask: Optional[Tensor] = None,
        mask_cross: Optional[Tensor] = None,
        numv: Optional[Tensor] = None,
        a: int = 10,
        b: int = 10,
        c: int = 10,
        temperature: float = 0.1
    ) -> Tensor:
        return self.mab(evectors, x, origin_x, graph, mask, mask_cross, numv, a, b, c, temperature)
    

class GraphMultisetTransformer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        evec_channels: int,
        Conv: Optional[Type] = None,
        num_nodes: int = 300,
        pool_sequences: List[str] = ['GMPool_G', 'SelfAtt', 'GMPool_I'],
        num_heads: int = 4,
        layer_norm: bool = False,
        args = None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.Conv = Conv or GCNConv
        self.num_nodes = num_nodes
        #self.pooling_ratio = pooling_ratio
        self.pool_sequences = pool_sequences
        self.num_heads = num_heads
        self.layer_norm = layer_norm
        self.args = args
        self.device = "cuda:" + args.device

        self.lin2_bp = Linear(in_channels, out_channels)
        self.lin2_mf = Linear(in_channels, out_channels)
        self.lin2_cc = Linear(in_channels, out_channels)

        self.pools = torch.nn.ModuleList()
        num_out_nodes = 30
        for i, pool_type in enumerate(pool_sequences):
            if pool_type not in ['GMPool_G', 'GMPool_I', 'SelfAtt', 'GMPool_evec']:
                raise ValueError("Elements in 'pool_sequences' should be one "
                                 "of 'GMPool_G', 'GMPool_I', or 'SelfAtt'")

            if i == len(pool_sequences) - 1:
                num_out_nodes = 1

            if pool_type == 'GMPool_G':
                self.pools.append(
                    PMA(hidden_channels, num_heads, num_out_nodes,
                        Conv=self.Conv, layer_norm=layer_norm))
                num_out_nodes = 30

            elif pool_type == 'GMPool_evec':
                self.a = num_nodes // 6
                self.c = self.a * 3
                self.b = num_nodes - self.a - self.c
                num_out_nodes = self.a + self.b + self.c
                self.pools.append(
                    GraphMA(in_channels, evec_channels, num_heads,
                        Conv=self.Conv, layer_norm=layer_norm))

            elif pool_type == 'GMPool_I':
                self.pools.append(
                    PMAGroup(in_channels, num_heads, num_out_nodes, Conv=None,
                        layer_norm=layer_norm))
                num_out_nodes = 1

            elif pool_type == 'SelfAtt':
                self.pools.append(
                    SAB(in_channels, in_channels, num_heads, Conv=None,
                        layer_norm=layer_norm))

    def reset_parameters(self):
        self.lin2_bp.reset_parameters()
        init.zeros_(self.lin2_bp.bias)
        self.lin2_mf.reset_parameters()
        init.zeros_(self.lin2_mf.bias)
        self.lin2_cc.reset_parameters()
        init.zeros_(self.lin2_cc.bias)
        for pool in self.pools:
            pool.reset_parameters()


    def forward(self, x: Tensor, origin_x: Tensor, evectors: Tensor, batch: Tensor,
                edge_index: Optional[Tensor] = None, numv: Tensor = None, temperature: float = 0.1) -> Tensor:
        """"""
        temp_batch = batch.clone()
        temp_batch = temp_batch.repeat_interleave(repeats=3)
        batch_x, mask = to_dense_batch(x, temp_batch)
        batch_origin_x, mask_origin_x = to_dense_batch(origin_x, temp_batch)
        mask = (~mask).unsqueeze(1).to(dtype=x.dtype) * -1e9

        a, b, c = self.a, self.b, self.c

        differences = [numv[i] - numv[i - 1] for i in range(1, len(numv))]
        mask_size = torch.max(torch.stack(differences))
        mask_cross = torch.zeros(len(numv) - 1, a + b + c, mask_size * 3, dtype=torch.bool).to(self.device)
        atob = a + b
        atoc = atob + c
        for i in range(len(numv) - 1):
            Ni = numv[i + 1] - numv[i]
            start_idx = 3 * numv[i]

            mask_cross[i, :a, 0 : Ni] = True
            mask_cross[i, a:atob, Ni : 2 * Ni] = True
            mask_cross[i, atob:atoc, 2 * Ni : 3 * Ni] = True
        mask_cross = (~mask_cross).to(dtype=x.dtype) * -1e9

        decorrelation_loss = 0
        for i, (name, pool) in enumerate(zip(self.pool_sequences, self.pools)):
            graph = (x, edge_index, batch) if name == 'GMPool_G' else None
            if name == 'GMPool_evec':
                batch_x, decol_loss = pool(batch_x, batch_origin_x, evectors, graph, mask, mask_cross, numv, a, b, c, temperature)
                decorrelation_loss += decol_loss
            elif name == "SelfAtt":
                batch_x = pool(batch_x, graph, mask, temperature)
            else:
                batch_x_bp, batch_x_mf, batch_x_cc = pool(batch_x, graph, mask, temperature)
            mask = None
        return self.lin2_bp(batch_x_bp.squeeze(1)), self.lin2_mf(batch_x_mf.squeeze(1)), self.lin2_cc(batch_x_cc.squeeze(1)), decorrelation_loss


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, pool_sequences={self.pool_sequences})')
