import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        """
        
        DEFINE YOUR NETWORK HERE
        
        """

        # change this obviously!
        self.naive = nn.Conv2d(in_channels=1, out_channels=num_classes, kernel_size=112, stride=1, padding=0, bias=True)
        
    def forward(self, x):
        """

        DEFINE YOUR FORWARD PASS HERE

        """

        # change this obviously!
        out = self.naive(x)
        
        return out