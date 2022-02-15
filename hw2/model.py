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
    def createLayer(self):
        return nn.Conv2d(self.C_in, self.C_out, self.kernel_size, \
            self.stride, self.padding, 1, self.groups, self.bias) 

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        """
        
        DEFINE YOUR NETWORK HERE
        
        """

        self.leaky_relu = nn.LeakyReLU(0.01) 
        self.max_pool = nn.MaxPool2d((2, 2), 2)

        conf1 = Conv2dConfig((12, 1, 112, 112), 8, (7, 7), 1, 0) 
        self.conv_1 = conf1.createLayer() 
        self.layer_norm_1 = nn.LayerNorm((conf1.C_out, conf1.H_out, conf1.W_out))
        
        conf2 = Conv2dConfig((12, conf1.C_out, int(conf1.H_out / 2), int(conf1.W_out / 2)), 8, (7, 7), 2, 0, 8)
        self.conv_2 = conf2.createLayer()
        self.layer_norm_2 = nn.LayerNorm((conf2.C_out, conf2.H_out, conf2.W_out))

        conf3 = Conv2dConfig((12, conf2.C_out, int(conf2.H_out / 2), int(conf2.W_out / 2)), 16, (1, 1), 1, 0)
        self.conv_ptwise_1 = conf3.createLayer()

        conf4 = Conv2dConfig((12, conf3.C_out, conf3.H_out, conf3.W_out), 16, (7, 7), 1, 0, 16)
        self.conv_dwise = conf4.createLayer() 
        self.layer_norm_3 = nn.LayerNorm((conf4.C_out, conf4.H_out, conf4.W_out))

        conf5 = Conv2dConfig((12, conf4.C_out, conf4.H_out, conf4.W_out), 32, (1, 1), 1, 0)
        self.conv_ptwise_2 = conf5.createLayer()

        conf6 = Conv2dConfig((12, conf5.C_out, int(conf5.H_out / 2), int(conf5.W_out / 2)), \
            num_classes, (int(conf5.H_out / 2), int(conf5.W_out / 2)), 1, 0, 1, True)
        self.fc = conf6.createLayer()
        
    def forward(self, x):
        """

        DEFINE YOUR FORWARD PASS HERE

        """
        tmp = self.max_pool(self.leaky_relu(self.layer_norm_1(self.conv_1(x))))
        tmp = self.conv_ptwise_1(self.max_pool(self.leaky_relu(self.layer_norm_2(self.conv_2(tmp)))))
        out = self.fc(self.conv_ptwise_2(self.max_pool(self.leaky_relu(self.layer_norm_3(self.conv_dwise(tmp))))))
        return out