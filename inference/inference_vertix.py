"""Utility for inference using trained networks"""

import torch

from data.shapenet import ShapeNet
from model.vertix_model import VertixModel


class InferenceHandlerVertixModel:
    """Utility for inference using trained PointNet network"""

    def __init__(self, ckpt, num_vertices):
        """
        :param ckpt: checkpoint path to weights of the trained network
        """
        self.model = VertixModel(num_vertices)
        self.model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        self.model.eval()

    def infer_single(self, inp):
        """
        Infer class of the shape given its point cloud representation
        """
        input_tensor = torch.from_numpy(inp).float().unsqueeze(0)

        # TODO: Predict class
        prediction = self.model(input_tensor)[0].detach().numpy().squeeze()

        return prediction