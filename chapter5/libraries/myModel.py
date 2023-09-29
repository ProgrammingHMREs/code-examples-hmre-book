#!/usr/bin/env python3

## Skeleton with a simple example for programming a model with PyTorch

import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(10, 5)

# Create model and move it to GPU
model = MyModel().cuda()

