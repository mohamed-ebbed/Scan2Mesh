import torch

from data.shapenet import ShapeNet
from model.vertix_edge_model import VertixEdge


class InferenceHandlerVertixEdgeModel:

    def __init__(self, ckpt, num_vertices, feature_size, device):

        self.device=device
        self.model = VertixEdge(num_vertices, feature_size).requires_grad_(False)
        self.model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        self.model.to(self.device)
        self.model.eval()

    def infer_single(self, inp, mask, v1s_idx, v2s_idx, adj):

        input_tensor = torch.from_numpy(inp).float().unsqueeze(0).to(self.device)
        adj = torch.from_numpy(adj).float().unsqueeze(0).to(self.device)
        mask = torch.from_numpy(mask).float().unsqueeze(0).to(self.device)

        vertices, edges = self.model(input_tensor, mask, v1s_idx, v2s_idx, adj)

        if self.device == "cpu":
            return vertices.detach().numpy().squeeze(), edges.argmax(1).numpy().squeeze()
        else:
            return vertices.detach().cpu().numpy().squeeze(), edges.argmax(1).cpu().numpy().squeeze()
