import torch.nn as nn 
import torch
from timm.loss import LabelSmoothingCrossEntropy


class Distillation_Loss(nn.Module):
    def __init__(self,) :
        super(Distillation_Loss,self).__init__()   
        self.cross_entropy = LabelSmoothingCrossEntropy()
        self.mse = nn.MSELoss()


    def forward(self,output,target,features_old=None,features_new=None,distilled_old_feats=None,
                distilled_new_feats=None):
        if features_old is None:
            loss = self.cross_entropy(output,target)
        elif 0 in distilled_new_feats or 0 in distilled_old_feats:
            loss = self.cross_entropy(output,target) + self.mse(features_new,features_old)
        else:
            loss = self.cross_entropy(output,target) + self.mse(features_new,features_old)+self.mse(distilled_new_feats,distilled_old_feats)
        # else:
        #     loss = self.cross_entropy(output,target) + self.mse(features_new,features_old)
        return loss
        