import torch
import torch.nn as nn
import numpy as np

class DistillationModel(nn.Module):
    def __init__(self,model, num_classes):
        
        super().__init__()
        self.model = model
        if hasattr(self.model.head,"in_features"):
            self.n_inputs = model.head.in_features
        else:
            self.n_inputs = model.head[0].in_features
        self.model.head = nn.Sequential(
            nn.Linear(self.n_inputs,512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512,num_classes) #!!!Output size must match number of labels!!!!
        )

    def forward(self,x,patches):
        features = self.model.forward_features(x)
        distilled_features = torch.empty((features.shape[0],len(patches),features.shape[2]))
        for patch in patches:
            distilled_features = torch.index_select(features,1,patch)
        return self.model(x), features ,distilled_features