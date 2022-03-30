import torch.nn as nn
import torch
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(
        self,
        args,
        dropout_prob=0.2,
    ):
        super(Decoder, self).__init__()

        # **** YOU SHOULD IMPLEMENT THE MODEL ARCHITECTURE HERE ****
        # Define the network architecture based on the figure shown in the assignment page.
        # Read the instruction carefully for layer details.
        # Pay attention that your implementation should include FC layers, weight_norm layers,
        # Relu layers, Dropout layers and a tanh layer.
        self.relu = nn.ReLU()
        self.dropout_prob = dropout_prob
        self.th = nn.Tanh()

        self.fully_connected_layers = [
            nn.Sequential(
                nn.utils.weight_norm(nn.Linear(3, 512)), 
                nn.ReLU(), 
                nn.Dropout(self.dropout_prob)
            ) for i in range(0, 7)
        ] 

        self.fc_reduce = nn.Linear(512, 1)

        # ***********************************************************************

    # input: N x 3
    def forward(self, input):
        # **** YOU SHOULD IMPLEMENT THE FORWARD PASS HERE ****
        # Based on the architecture defined above, implement the feed forward procedure
        output = self.fully_connected_layers[0](input) 
        for i, fc in enumerate(self.fully_connected_layers[1:]):
            output = fc(output)
            if i == 3: 
                output = torch.cat((output, input), dim=0)

        return self.th(self.fc_reduce(output))
        
        # ***********************************************************************
