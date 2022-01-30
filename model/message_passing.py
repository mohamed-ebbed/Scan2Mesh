import torch

import torch.nn as nn

class NodeToEdge(nn.Module):
    def __init__(self):
        super(NodeToEdge, self).__init__()

    def forward(self, hv, v1s_idx, v2s_idx):

        batch_size, num_vertices, _ = hv.shape

        v1s = hv[:,v1s_idx]

        v2s = hv[:,v2s_idx]

        vertices_concat = torch.cat([v1s,v2s], axis=-1).reshape(batch_size, num_vertices, num_vertices, -1)

        return vertices_concat

class NodeToEdgeTriple(nn.Module):
    def __init__(self):
        super(NodeToEdgeTriple, self).__init__()

    def forward(self, hv, v1s_idx, v2s_idx, v3d_idx):

        batch_size, num_vertices, _ = hv.shape

        v1s = hv[:,v1s_idx]

        v2s = hv[:,v2s_idx]

        v3s = hv[:,v3s_idx]

        vertices_concat = torch.cat([v1s,v2s,v3s], axis=-1).reshape(batch_size, num_vertices, num_vertices, num_vertices, -1)

        return vertices_concat

class EdgeToNode(nn.Module):
    def __init__(self):
        super(EdgeToNode, self).__init__()

    def forward(self, he, adj):

        hv = torch.sum(he*adj, axis=-2)
        
        return hv

class EdgeToNodeTriple(nn.Module):
    def __init__(self):
        super(EdgeToNodeTriple, self).__init__()

    def forward(self, he, adj):

        hv = torch.sum(he*adj, dim=(-2,-3))
        
        return hv

