import torch

from data.shapenet import ShapeNet
from model.vertix_edge_model import VertixEdge


class InferenceHandlerVertixEdgeModel:

    def __init__(self, ckpt, num_vertices, feature_size):

        self.model = VertixEdge(num_vertices, feature_size)
        self.model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        self.model.eval()

    def infer_single(self, inp, v1s_idx, v2s_idx, adj):

        input_tensor = torch.from_numpy(inp).float().unsqueeze(0)
        adj = torch.from_numpy(adj).float().unsqueeze(0)

        vertices, edges = self.model(input_tensor, v1s_idx, v2s_idx, adj)

        return vertices.detach().numpy().squeeze(), edges.argmax(1).numpy().squeeze()