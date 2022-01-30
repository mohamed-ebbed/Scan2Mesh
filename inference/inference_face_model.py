import torch

from data.shapenet import ShapeNet
from model.face_model import FaceModel


class InferenceHandlerFaceModel:

    def __init__(self, ckpt, num_vertices, feature_size, device):

        self.device=device
        self.model = FaceModel(feature_size).requires_grad_(False)
        self.model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        self.model.to(self.device)
        self.model.eval()

    def infer_single(self, hv, adj):

        hv = torch.from_numpy(hv).float().unsqueeze(0).to(self.device)
        adj = torch.from_numpy(adj).float().to(self.device).long()

        faces = self.model(hv, adj).argmax(1)

        if self.device == "cpu":
            return faces.detach().numpy().squeeze()
        else:
            return faces.detach().cpu().numpy().squeeze()
