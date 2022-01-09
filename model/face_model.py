import torch 

import torch.nn as nn

from message_passing import *


class FaceModel(nn.Module):
    def __init__(self, feature_size, v1s_idx, v2s_idx, adj):
        super(FaceModel, self).__init__()

        self.feature_size = feature_size

        self.mlp_v = nn.Sequential(
            nn.Linear(feature_size,feature_size),
            nn.ELU(),
            nn.Linear(feature_size,feature_size),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(feature_size)
        )


        self.stage1 = nn.Sequential(
            nn.Linear(feature_size,feature_size),
            nn.ELU(),
            nn.Linear(feature_size,feature_size),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(feature_size)
        )

        self.stage2= nn.Sequential(
            NodeToEdge(v1s_idx, v2s_idx),
            nn.Linear(feature_size,feature_size),
            nn.ELU(),
            nn.Linear(feature_size,feature_size),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(feature_size),
            EdgeToNode(adj),
            nn.Linear(feature_size,feature_size),
            nn.ELU(),
            nn.Linear(feature_size,feature_size),
            nn.ELU(),
            nn.Dropout(0.5),

        )

        self.stage3 = nn.Sequential(
            nn.Linear(2*feature_size,feature_size),
            nn.ELU(),
            nn.Linear(feature_size,feature_size),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(feature_size),
            nn.Linear(feature_size,2)
        )


    def forward(self):


        #TODO preprocessing

        edge_features = ...

        stage1_out = self.stage1(edge_features)

        stage2_out = self.stage2(stage1_out)

        edge_f_concat = torch.cat([stage1_out, stage2_out], axis=-1)

        out = self.stage3(edge_f_concat)

        return out
