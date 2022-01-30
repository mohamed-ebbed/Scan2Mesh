import torch 

import torch.nn as nn

from model.message_passing import *

import torch_geometric.nn as gnn


class FaceModel(nn.Module):
    def __init__(self, face_feature_size):
        super(FaceModel, self).__init__()

        self.feature_size = face_feature_size

        self.mlp_face = nn.Sequential(
            nn.Linear(7,self.feature_size),
            nn.ELU(),
            nn.Linear(self.feature_size,self.feature_size),
            nn.ELU(),
        )

        self.gnn1 = gnn.GraphConv(self.feature_size,self.feature_size)
        self.gnn2 = gnn.GraphConv(self.feature_size,self.feature_size)


        self.stage1 = nn.Sequential(
            nn.ELU(),
            nn.Linear(self.feature_size,self.feature_size),
            nn.ELU(),
            nn.Linear(self.feature_size,self.feature_size),
            nn.ELU(),
            nn.Linear(self.feature_size,self.feature_size),
            nn.ELU()
        )

        self.stage2 = nn.Sequential(
            nn.Linear(2*self.feature_size,self.feature_size),
            nn.ELU(),
            nn.Linear(self.feature_size,self.feature_size),
            nn.ELU(),
            nn.Linear(self.feature_size,2)
        )


    def forward(self, hv, adj):


        face_features = self.mlp_face(hv)

        stage1_in = self.gnn1(face_features, adj)

        stage1_out = self.stage1(stage1_in)

        stage1_out = self.gnn2(stage1_out, adj)

        edge_f_concat = torch.cat([face_features, stage1_out], axis=-1)

        out = self.stage2(edge_f_concat).transpose(-1,1)

        return out
