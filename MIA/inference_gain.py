import torch
import torch.nn as nn

class NegetiveInferenceGain(nn.Module):
    def __init__(self, beta=0.1):
        super(NegetiveInferenceGain, self).__init__()
        self.beta = beta
        self.bce = nn.BCELoss()
    def forward(self,output, target):
        return self.beta*self.bce(output,target)