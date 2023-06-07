from transformers import ViTModel, ViTConfig
import torch.nn as nn
from train import model_checkpoint
from tqdm import tqdm
import torch


class ViT(nn.Module):

  def __init__(self, config=ViTConfig(), num_labels=500, 
               model_checkpoint=model_checkpoint):

        super(ViT, self).__init__()

        self.vit = ViTModel.from_pretrained(model_checkpoint, add_pooling_layer=False)
        self.classifier = (
            nn.Linear(config.hidden_size, num_labels) 
        )

  def forward(self, x):
      
    x = self.vit(x)['last_hidden_state']
    # Use the embedding of [CLS] token
    output = self.classifier(x[:, 0, :])

    return output
  

