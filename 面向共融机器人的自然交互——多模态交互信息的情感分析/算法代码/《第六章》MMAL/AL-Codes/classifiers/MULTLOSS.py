import torch
import torch.nn as nn 
import torch.nn.functional as F 


class MULTLOSS(nn.Module):
    def __init__(self, args, feature_sizes=[256, 64, 32], interm_dim=128):
        super(MULTLOSS, self).__init__()
        
        # self.GAP1 = nn.AvgPool2d(feature_sizes[0])
        # self.GAP2 = nn.AvgPool2d(feature_sizes[1])
        # self.GAP3 = nn.AvgPool2d(feature_sizes[2])

        self.FC1 = nn.Linear(feature_sizes[0], interm_dim)
        self.FC2 = nn.Linear(feature_sizes[1], interm_dim)
        self.FC3 = nn.Linear(feature_sizes[2], interm_dim)
        # self.FC4 = nn.Linear(num_channels[3], interm_dim)

        self.linear = nn.Linear(3 * interm_dim, 1)
    
    def forward(self, features):
        out1 = F.relu(self.FC1(features[0]))
        out2 = F.relu(self.FC2(features[1]))
        out3 = F.relu(self.FC3(features[2]))
        out = self.linear(torch.cat((out1, out2, out3), 1))
        return out