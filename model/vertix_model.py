import torch

import torch.nn as nn

class VertixModel(nn.Module):
    def __init__(self, output_size):
        super(VertixModel, self).__init__()
        self.output_size = output_size

        self.conv_layers_1 = nn.Sequential(
            nn.Conv3d(5,16, 4, 2, 0),
            #nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16,16, 1, 1, 0),
            #nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16,32, 3, 2, 0),
            #nn.BatchNorm3d(32),
            nn.ReLU(inplace=True), 
            nn.Conv3d(32,32, 1, 1, 0),
            #nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )

        self.conv_layers_2 = nn.Sequential(
            nn.Conv3d(32,64, 3, 2, 0),
            #nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64,64, 1, 1, 0),
            #nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64,256, 3, 1, 0),
            #nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(256,256),
            #nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256,self.output_size*3)
        )

            

    def forward(self, x):

        x = self.conv_layers_1(x)
        f2 = x
        x = self.conv_layers_2(x)
        x = x.reshape(-1,256)
        x = self.linear_layers(x)
        x = x.reshape(-1,self.output_size, 3)

        return x, f2