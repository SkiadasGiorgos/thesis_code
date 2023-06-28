from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from create_train_lists import create_distil_dataset
import timm
import pickle
from timm.loss import LabelSmoothingCrossEntropy
import copy
import os
import torchvision
from model import DistillationModel

model_save_path = "/home/skiadasg/thesis_code/thesis_code/results/test_model.pkl"

seed = 42
lr = 1e-4
epochs = 2
batch_size = 32

old_classes_number=500
new_classes_number=0

train_ds, eval_ds= create_distil_dataset()  

train_dl = DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=32)
eval_dl = DataLoader(eval_ds,batch_size=batch_size,shuffle=True,num_workers=32)

def dataloaders(phase,train_dl=train_dl,eval_dl=eval_dl):
  if phase == "train":return train_dl
  elif phase == "val":
    return eval_dl

def dataset_size(phase,train_ds=train_ds,eval_ds=eval_ds):
    if phase == "train":
      return len(train_ds)
    elif phase == "val":
      return len(eval_ds)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
   
model_checkpoint = 'vit_small_r26_s32_224.augreg_in21k_ft_in1k'

model = timm.create_model(model_checkpoint,pretrained=True,num_classes=new_classes_number+old_classes_number)

distil_model = DistillationModel(model,old_classes_number+new_classes_number)
criterion = LabelSmoothingCrossEntropy()
criterion=criterion.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr,)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=3, gamma=0.97)

# n_inputs = model.head.in_features
# model.head = nn.Sequential(
#     nn.Linear(n_inputs,512),
#     nn.ReLU(),
#     nn.Dropout(0.3),
#     nn.Linear(512,old_classes_number+new_classes_number) #!!!Output size must match number of labels!!!!
# )

model.to(device)

def train_model(model,optimizer,scheduler,num_epochs):
  
  metrics = {"train_acc":[],"val_acc":[],"train_loss":[],"val_loss":[]}
  for epoch in range(num_epochs):
    print(f'Epoch:',epoch)
    print('-'*10) 

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for phase in ["train","val"]:
      if phase == "train":  
        model.train()
      else:
        pass
        model.eval()
      running_loss  = 0.0
      running_corrects = 0.0
      for x_train,y_train in tqdm(dataloaders(phase)):
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        
        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == "train"):
          outputs  = distil_model.forward(x_train)
          _,preds = torch.max(outputs,1)

          loss = criterion(outputs,y_train)
          
          if phase  == "train":    
            loss.backward()
            optimizer.step()
          running_loss += loss.item()*x_train.size(0)
          running_corrects += torch.sum(preds == y_train.data)
          
      if phase == "train": 
        scheduler.step()


      epoch_loss = running_loss/dataset_size(phase)
      epoch_acc = running_corrects.double()/dataset_size(phase)

        
      print("Loss: {:.4f} Acc: {:.4f}".format(epoch_loss, epoch_acc))


      if  phase == "val"  and epoch_acc>best_acc:
        best_model_wts=copy.deepcopy(model.state_dict())
        best_model = model

      if phase == "train":
        metrics["train_loss"].append(epoch_loss)
        metrics["train_acc"].append(epoch_acc.item())
      elif phase == "val":
        metrics["val_loss"].append(epoch_loss)
        metrics["val_acc"].append(epoch_acc.item())

    # results.append(metrics)
      
    # with open(save_path, "wb") as h:
    #   pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

  
  torch.save(best_model,model_save_path)
  return model.load_state_dict(best_model_wts)


model = train_model(model=model,optimizer=optimizer,scheduler=exp_lr_scheduler,num_epochs=epochs)
