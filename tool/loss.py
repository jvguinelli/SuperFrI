import torch
import torch.nn as nn

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss
    

class CustomMSELoss(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.mask = mask.reshape([1,1, 1, 7, 7])
        
    def forward(self, yhat, y):
        squared_diff = (yhat - y) ** 2
        masked_squared_diff = squared_diff * self.mask
        loss = masked_squared_diff.sum() / self.mask.sum()
        return loss