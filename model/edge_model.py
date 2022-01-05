import torch 

import torch.nn as nn


class MessagePassing(nn.Module):
    def __init__(self, input_dim, output_dim):

        super(self, MessagePassing).__init__()

        self.vertix_to_edge = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1(output_dim)
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1(output_dim)
            nn.ReLU()
        )

        self.edge_to_vertix = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1(output_dim)
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1(output_dim)
            nn.ReLU()
        )


    def forward(adj, v_h, e_h):
        
