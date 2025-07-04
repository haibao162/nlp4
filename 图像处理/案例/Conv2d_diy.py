import torch
import torch.nn as nn
import numpy as np

"""
手动实现简单的神经网络
使用pytorch实现CNN
手动实现CNN
对比
"""

class TorchCNN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel):
        super(TorchCNN, self).__init__()
        self.layer = nn.Conv2d(in_channel, out_channel, kernel, bias=False)
        # print(self.layer.state_dict()['weight'].shape, 'layer.shape')
        # torch.Size([5, 1, 2, 2]) layer.shape
    
    def forward(self, x):
        return self.layer(x)
    

x = np.array([[0.1, 0.2, 0.3, 0.4],
              [-3, -4, -5, -6],
              [5.1, 6.2, 7.3, 8.4],
              [-0.7, -0.8, -0.9, -1]])
    
in_channel = 1
out_channel = 5
kernel_size = 2
torch_model = TorchCNN(in_channel, out_channel, kernel_size)
torch_x = torch.FloatTensor([[x]])
print(torch_x.shape, 'torch_x.shape')
# torch.Size([1, 1, 4, 4])
output = torch_model.forward(torch_x)
output = output.detach().numpy()
print(output, output.shape, "torch模型预测结果\n")
# (1, 3, 3, 3)

class DiyModel:
    def __init__(self, input_height, input_width, weights, kernel_size):
        self.height = input_height
        self.width = input_width
        self.weights = weights
        self.kernel_size = kernel_size

    def forward(self, x):
        output = []
        for kernel_weight in self.weights:
            kernel_weight = kernel_weight.squeeze().numpy() #shape : 2x2


