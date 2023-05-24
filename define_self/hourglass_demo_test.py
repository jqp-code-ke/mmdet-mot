from mmdet.models import HourglassNet
import torch
self = HourglassNet()
self.eval()
inputs = torch.rand(1, 9, 256, 256)
level_outputs = self.forward(inputs)
for level_output in level_outputs:
    print(tuple(level_output.shape))
