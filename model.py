import torch
import torch.nn as nn

class DistillationModel(nn.Module):
    def __init__(self,model, num_classes):
        
        super().__init__()
        self.model = model
        self.n_inputs = model.head.in_features
        self.model.head = nn.Sequential(
            nn.Linear(self.n_inputs,512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512,num_classes) #!!!Output size must match number of labels!!!!
        )

    def forward(self,x):
        return self.model(x)