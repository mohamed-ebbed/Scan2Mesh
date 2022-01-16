import torch.nn as nn

from model.edge_model_simple import *

from model.vertix_model import *


class VertixEdge(nn.Module):
    def __init__(self, num_vertices, feature_size):
        super(VertixEdge, self).__init__()

        self.vertix_model = VertixModel(num_vertices)

        self.edge_model = EdgeModelSimple(feature_size)

    def forward(self, x, mask, v1s_idx, v2s_idx, adj):
        vertices, f2 = self.vertix_model(x)

        return vertices, self.edge_model(vertices,f2, v1s_idx, v2s_idx, adj)