import torch.nn as nn

IMAGE_WIDTH = 100 
IMAGE_HEIGHT = 100

class MaskedCNN(nn.Conv2d):
    """
    Masked convolution as explained in the PixelCNN variant of
    Van den Oord et al, “Pixel Recurrent Neural Networks”, NeurIPS 2016
    It inherits from Conv2D (uses the same parameters, plus the option to select a mask including
    the center pixel or not, as described in class and in the Fig. 2 of the above paper)
    """

    def __init__(self, mask_type, *args, **kwargs):
        self.mask_type = mask_type
        super(MaskedCNN, self).__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, height, width = self.weight.size()
        self.mask.fill_(1)
        if mask_type == 'A':
            self.mask[:, :, height//2, width//2:] = 0
            self.mask[:, :, height//2+1:, :] = 0
        else:
            self.mask[:, :, height//2, width//2+1:] = 0
            self.mask[:, :, height//2+1:, :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedCNN, self).forward(x)


class PixelCNN(nn.Module):
    """
    A PixelCNN variant you have to implement according to the instructions
    """

    def __init__(self):
        super(PixelCNN, self).__init__()

        # WRITE CODE HERE TO IMPLEMENT THE MODEL STRUCTURE

        self.relu = nn.LeakyReLU(0.001)

        # TODO: check if you need to use bias! 
        self.conv1 = MaskedCNN('A', in_channels=1, out_channels=16, \
            padding=(3, 3), kernel_size=(3, 3), stride=1, dilation=3, padding_mode='reflect')
        self.batch_norm1 = nn.BatchNorm2d(16)

        self.conv2 = MaskedCNN('B', in_channels=16, out_channels=16, \
            padding=(3, 3), kernel_size=(3, 3), stride=1, dilation=3, padding_mode='reflect')
        self.batch_norm2 = nn.BatchNorm2d(16)

        self.conv3 = MaskedCNN('B', in_channels=16, out_channels=16, \
            padding=(3, 3), kernel_size=(3, 3), stride=1, dilation=3, padding_mode='reflect')
        self.batch_norm3 = nn.BatchNorm2d(16)

        self.conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        block_output1 = self.relu(self.batch_norm1(self.conv1(x)))
        block_output2 = self.relu(self.batch_norm2(self.conv2(block_output1)))
        block_output3 = self.relu(self.batch_norm3(self.conv3(block_output2)))
        grayscale_intensity = self.conv(block_output3)
        return self.sigmoid(grayscale_intensity)
