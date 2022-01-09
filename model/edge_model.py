import torch 

import torch.nn as nn

from model.message_passing import *


class EdgeModel(nn.Module):
    def __init__(self, feature_size, v1s_idx, v2s_idx, adj):
        super(EdgeModel, self).__init__()

        self.feature_size = feature_size

        self.mlp_v = nn.Sequential(
            nn.Linear(3,feature_size),
            nn.ELU(),
            nn.Linear(feature_size,feature_size),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(feature_size)
        )

        self.mlp_f2 = nn.Sequential(
            nn.Linear(32*7*7*7,feature_size),
            nn.ELU(),
            nn.Linear(feature_size,feature_size),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(feature_size)
        )



        self.stage1 = nn.Sequential(
            NodeToEdge(v1s_idx, v2s_idx),
            nn.Linear(feature_size*2,feature_size),
            nn.ELU(),
            nn.Linear(feature_size,feature_size),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(feature_size)
        )

        self.stage2= nn.Sequential(
            EdgeToNode(adj),
            nn.Linear(feature_size,feature_size),
            nn.ELU(),
            nn.Linear(feature_size,feature_size),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(feature_size),
            NodeToEdge(v1s_idx, v2s_idx)

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


    def forward(self, vertices, f2):


        batch_size, num_vertices, _ = vertices.shape 

        vertices = vertices.reshape(-1,3)

        indices = (vertices + 1) * 6

        
        xs = indices[:,0].squeeze().long()
        ys = indices[:,1].squeeze().long()
        zs = indices[:,2].squeeze().long()

        f2_extracted = f2[:,xs,ys,zs,:]

        v_features = self.mlp_f1(vertices)
        f2_features = self.mlp_f2(f2_extracted)

        vertix_features = torch.cat([v_features,f2_features], axis=-1)

        vertix_features = vertix_features.reshape(batch_size, num_vertices, -1)

        stage1_out = self.stage1(vertix_features)

        stage2_out = self.stage2(stage1_out)

        edge_f_concat = torch.cat([stage1_out, stage2_out], axis=-1).reshape(batch_size*num_vertices*num_vertices,-1)

        out = self.stage3(edge_f_concat).reshape(batch_size, num_vertices, num_vertices, 2)

        return out








        
