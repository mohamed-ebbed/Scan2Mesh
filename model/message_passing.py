import torch.nn as nn

class NodeToEdge(nn.Module):
    def __init__(self, v1s_idx, v2s_idx):
        super(NodeToEdge, self).__init__()
        self.v1s_idx = v1s_idx
        self.v2s_idx = v2s_idx

    def forward(self, hv):

        batch_size, num_vertices, _ = hv.shape

        v1s = hv[:,self.v1s_idx]

        v2s = hv[:,self.v2s_idx]

        vertices_concat = torch.cat([v1s,v2s], axis=-1).reshape(batch_size, num_vertices, num_vertices, -1)

        return vertices_concat

class EdgeToNode(nn.Module):
    def __init__(self, adj):
        super(EdgeToNode, self).__init__()
        self.adj = adj

    def forward(self, he):

        hv = torch.sum(he*self.adj, axis=-2)
        
        return hv
