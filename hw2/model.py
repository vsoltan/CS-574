import torch
import torch.nn as nn

def conv_out_size(in_size, receptive_field_size, stride, padding):
        return 1 + int((in_size - receptive_field_size + 2 * padding) / stride)

class Conv2dConfig():
    def __init__(self, input_size, out_channels, kernel_size, stride, padding, groups=1, bias=False):
        self.N, self.C_in, self.H, self.W = input_size 
        self.kernel_size, self.C_out = kernel_size, out_channels
        self.stride, self.padding = stride, padding 

        self.H_out = conv_out_size(self.H, self.kernel_size[0], \
            self.stride, self.padding)

        self.W_out = conv_out_size(self.W, self.kernel_size[1], \
            self.stride, self.padding)

        self.groups, self.bias = groups, bias 

    ''' created layer takes input with size (N, C_in, H, W) and produces output with size 
        (N, C_out, H_out, W_out)'''
    def createLayer(self, init_scheme):
        layer = nn.Conv2d(self.C_in, self.C_out, self.kernel_size, \
            self.stride, self.padding, 1, self.groups, self.bias)

        # initialize weights and biases
        if layer.bias is not None: 
            nn.init.constant_(layer.bias.data, 0)

        if init_scheme == 'kaiming':
            nn.init.kaiming_uniform_(layer.weight, 0.01)
        elif init_scheme == 'xavier':
            nn.init.xavier_uniform_(layer.weight)
        else:
            print("undefined weight initialization scheme, using default.")

        return layer  

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        """
        
        DEFINE YOUR NETWORK HERE
        
        """
        # data set parameters 
        num_views, in_channels, in_height, in_width = 12, 1, 112, 112

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01) 
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        conf1 = Conv2dConfig(input_size=(num_views, in_channels, in_height, in_width), out_channels=8, \
            kernel_size=(7, 7), stride=1, padding=0) 
        self.conv_1 = conf1.createLayer(init_scheme='kaiming') 
        self.layer_norm_1 = nn.LayerNorm((conf1.C_out, conf1.H_out, conf1.W_out))
        
        conf2 = Conv2dConfig(input_size=(num_views, conf1.C_out, int(conf1.H_out / 2), int(conf1.W_out / 2)), \
            out_channels=8, kernel_size=(7, 7), stride=2, padding=0, groups=8)
        self.conv_2 = conf2.createLayer(init_scheme='kaiming')
        self.layer_norm_2 = nn.LayerNorm((conf2.C_out, conf2.H_out, conf2.W_out))

        conf3 = Conv2dConfig(input_size=(num_views, conf2.C_out, int(conf2.H_out / 2), int(conf2.W_out / 2)), \
            out_channels=16, kernel_size=(1, 1), stride=1, padding=0, groups=1, bias=True)
        self.conv_ptwise_1 = conf3.createLayer(init_scheme='xavier')

        conf4 = Conv2dConfig(input_size=(num_views, conf3.C_out, conf3.H_out, conf3.W_out), out_channels=16, \
            kernel_size=(7, 7), stride=1, padding=0, groups=16)
        self.conv_dwise = conf4.createLayer(init_scheme='kaiming') 
        self.layer_norm_3 = nn.LayerNorm((conf4.C_out, conf4.H_out, conf4.W_out))

        conf5 = Conv2dConfig(input_size=(num_views, conf4.C_out, conf4.H_out, conf4.W_out), out_channels=32, \
            kernel_size=(1, 1), stride=1, padding=0, groups=1, bias=True)
        self.conv_ptwise_2 = conf5.createLayer(init_scheme='xavier')

        conf6 = Conv2dConfig(input_size=(num_views, conf5.C_out, int(conf5.H_out / 2), int(conf5.W_out / 2)), \
            out_channels=num_classes, kernel_size=(int(conf5.H_out / 2), int(conf5.W_out / 2)), stride=1, padding=0, \
            groups=1, bias=True)
        self.fc = conf6.createLayer(init_scheme='xavier')
        
    def forward(self, x):
        """

        DEFINE YOUR FORWARD PASS HERE

        """
        tmp = self.max_pool(self.leaky_relu(self.layer_norm_1(self.conv_1(x))))
        tmp = self.conv_ptwise_1(self.max_pool(self.leaky_relu(self.layer_norm_2(self.conv_2(tmp)))))
        out = self.fc(self.conv_ptwise_2(self.max_pool(self.leaky_relu(self.layer_norm_3(self.conv_dwise(tmp))))))
        return out