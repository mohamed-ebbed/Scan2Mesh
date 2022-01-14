import torch

from data.shapenet import ShapeNet
from model.vertix_model import VertixModel


class InferenceHandlerVertixModel:

    def __init__(self, ckpt, num_vertices):

        self.model = VertixModel(num_vertices)
        self.model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        self.model.eval()

    def infer_single(self, inp):

        input_tensor = torch.from_numpy(inp).float().unsqueeze(0)

        prediction = self.model(input_tensor)[0].detach().numpy().squeeze()

        return prediction