import torch.nn as nn
import torch
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        self.dropout = nn.Dropout(self.dropout_prob)
        self.th = nn.Tanh()

        self.fc1 = nn.utils.weight_norm(nn.Linear(3, 512)).to(device)
        self.fc2 = nn.utils.weight_norm(nn.Linear(512, 512)).to(device)
        self.fc3 = nn.utils.weight_norm(nn.Linear(512, 512)).to(device)
        self.fc4 = nn.utils.weight_norm(nn.Linear(512, 509)).to(device)
        self.fc5 = nn.utils.weight_norm(nn.Linear(512, 512)).to(device)
        self.fc6 = nn.utils.weight_norm(nn.Linear(512, 512)).to(device)
        self.fc7 = nn.utils.weight_norm(nn.Linear(512, 512)).to(device)
        self.fc_reduce = nn.Linear(512, 1).to(device)

        # ***********************************************************************

    # input: N x 3
    def forward(self, input):
        # **** YOU SHOULD IMPLEMENT THE FORWARD PASS HERE ****
        # Based on the architecture defined above, implement the feed forward procedure
        output = self.dropout(self.relu(self.fc1(input)))
        output = self.dropout(self.relu(self.fc2(output)))
        output = self.dropout(self.relu(self.fc3(output)))
        output = self.dropout(self.relu(self.fc4(output)))
        output = torch.cat((output, input), dim=1)
        output = self.dropout(self.fc5(output))
        output = self.dropout(self.fc6(output))
        output = self.dropout(self.fc7(output))
        output = self.fc_reduce(output)
        return self.th(output)
        # ***********************************************************************
