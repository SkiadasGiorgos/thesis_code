from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from create_train_lists import create_datasets
import timm
import pickle
from timm.loss import LabelSmoothingCrossEntropy
import copy
import os

model_checkpoint = "vit_large_r50_s32_224.augreg_in21k"

seed = 42
lr = 1e-4
epochs = 10
batch_size = 32
save_path = "/home/skiadasg/thesis_code/thesis_code/results/results.pkl"
model_save_path = "/home/skiadasg/thesis_code/thesis_code/results/resnet_model_vgg.pkl"

config = {
    "model_ckp": model_checkpoint,
    "dataset": "VggFace2",
    "seed": seed,
    "batch_size": batch_size,
    "lr": lr,
    "epochs": epochs   
}

train_data_path = "~/home/nas2/ckoutlis/DataStorage/vggface2/train"

train_ds, eval_ds, test_ds = create_datasets()  

train_dl = DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=32)
eval_dl = DataLoader(eval_ds,batch_size=batch_size,shuffle=True,num_workers=32)
test_dl = []

def dataloaders(phase,train_dl=train_dl,eval_dl=eval_dl,test_dl=test_dl):
  if phase == "train":return train_dl
  elif phase == "val":
    return eval_dl
  elif phase == "test": 
    return test_dl

def dataset_size(phase,train_ds=train_ds,eval_ds=eval_ds,test_ds=test_ds):
    if phase == "train":
      return len(train_ds)
    elif phase == "val":
      return len(eval_ds)
    elif phase == "test": 
      return len(test_ds)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

model = timm.create_model(model_checkpoint,pretrained=True,num_classes=0,)

for param in model.parameters():
  param.requires_grad =  False

param_name = [name for name,_ in model.named_parameters()] # All parameters name
layer_name = [name for name,_ in model.named_modules()] # All layers name


def debarcle_layers(model, num_debarcle):
    '''Debarcle From the last [-1]layer to the [-num_debarcle] layers, 
    approximately(for there is Conv2d which has only weight parameter)'''
    num_debarcle *= 2
    param_debarcle = param_name[-num_debarcle:]
    if param_debarcle[0].split('.')[-1] == 'bias':
        param_debarcle = param_name[-(num_debarcle + 1):]
    for name, param in model.named_parameters():
        param.requires_grad = True if name in param_debarcle else False

debarcle_layers(model,len(layer_name))


results = []
results.append(config)

def train_model(model):
  feature = None
  for x_train,y_train in tqdm(dataloaders("val")):
    # feats = model(x_train)
    feats = model.forward_features(x_train)
    print(feats.shape)
  return model.load_state_dict(feature)


model = train_model(model=model)

