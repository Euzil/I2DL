"""SegmentationNN"""
import torch
import torch.nn as nn

class ConvLayer(nn.Module):
    """Basic convolution block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class SegmentationNN(nn.Module):
    def __init__(self, num_classes=23, hp=None):
        super().__init__()
        ########################################################################
        #                             YOUR CODE                                  #
        ########################################################################

        # 基础通道数
        f = 48
        
        # 编码器
        self.e1 = ConvLayer(3, f)
        self.e2 = ConvLayer(f, f)
        self.pool1 = nn.MaxPool2d(2)
        
        self.e3 = ConvLayer(f, 2*f)
        self.e4 = ConvLayer(2*f, 2*f)
        self.pool2 = nn.MaxPool2d(2)
        
        self.e5 = ConvLayer(2*f, 4*f)
        self.e6 = ConvLayer(4*f, 4*f)
        self.pool3 = nn.MaxPool2d(2)
        
        # 桥接
        self.b1 = ConvLayer(4*f, 8*f)
        self.b2 = ConvLayer(8*f, 8*f)
        
        # 解码器
        self.upconv1 = nn.ConvTranspose2d(8*f, 4*f, 2, stride=2)
        self.d1 = ConvLayer(8*f, 4*f)
        self.d2 = ConvLayer(4*f, 4*f)
        
        self.upconv2 = nn.ConvTranspose2d(4*f, 2*f, 2, stride=2)
        self.d3 = ConvLayer(4*f, 2*f)
        self.d4 = ConvLayer(2*f, 2*f)
        
        self.upconv3 = nn.ConvTranspose2d(2*f, f, 2, stride=2)
        self.d5 = ConvLayer(2*f, f)
        self.d6 = ConvLayer(f, f)
        
        # 最终输出层
        self.final = nn.Conv2d(f, num_classes, 1)

        ########################################################################
        #                           END OF YOUR CODE                            #
        ########################################################################

    def forward(self, x):
        ########################################################################
        #                             YOUR CODE                                 #
        ########################################################################
        
        # 编码器部分
        e1 = self.e2(self.e1(x))
        e2 = self.e4(self.e3(self.pool1(e1)))
        e3 = self.e6(self.e5(self.pool2(e2)))
        
        # 桥接部分
        b = self.b2(self.b1(self.pool3(e3)))
        
        # 解码器部分（带跳跃连接）
        d1 = self.d2(self.d1(torch.cat([self.upconv1(b), e3], 1)))
        d2 = self.d4(self.d3(torch.cat([self.upconv2(d1), e2], 1)))
        d3 = self.d6(self.d5(torch.cat([self.upconv3(d2), e1], 1)))
        
        # 最终输出
        x = self.final(d3)
        
        return x

        ########################################################################
        #                           END OF YOUR CODE                            #
        ########################################################################

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

class DummySegmentationModel(nn.Module):
    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)
            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1
        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()

if __name__ == "__main__":
    from torchinfo import summary
    summary(SegmentationNN(), (1, 3, 240, 240), device="cpu")