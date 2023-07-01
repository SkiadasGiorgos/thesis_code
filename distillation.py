from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from create_train_lists import create_distil_dataset
import timm
import pickle
import copy
import numpy as np
from model import DistillationModel
from loss import Distillation_Loss

model_save_path = "/home/skiadasg/thesis_code/thesis_code/results/test_model.pkl"
teacher_model_path = "results/teacher_classes100.pkl"
seed = 42
lr = 1e-4
epochs = 3
batch_size = 32

old_classes_number=100
new_classes_number=50

train_ds, eval_ds= create_distil_dataset(old_classes_number=old_classes_number,new_classes_number=new_classes_number)  

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

# model_checkpoint = 'vit_small_r26_s32_224.augreg_in21k_ft_in1k'
# model = timm.create_model(model_checkpoint)

teacher_model =  torch.load(teacher_model_path)
teacher_model.eval()
distil_model = DistillationModel(teacher_model,old_classes_number+new_classes_number)
distil_loss = Distillation_Loss()
distil_loss = distil_loss.to(device)
optimizer = optim.Adam(distil_model.parameters(), lr=lr,)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=3, gamma=0.97)


distil_model.to(device)
teacher_model.to(device)

def train_model(distil_model,teacher_model,optimizer,scheduler,num_epochs):
  
  metrics = {"train_acc":[],"val_acc":[],"train_loss":[],"val_loss":[]}
  for epoch in range(num_epochs):
    print(f'Epoch:',epoch)
    print('-'*10) 

    best_acc = 0.0
    best_model_wts = copy.deepcopy(distil_model.state_dict())

    for phase in ["train","val"]:
      if phase == "train":  
        distil_model.train()
      else:
        distil_model.eval()
      running_loss  = 0.0
      running_corrects = 0.0


      for x_train,y_train,patches in tqdm(dataloaders(phase)):
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        patches = patches.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == "train"):
          outputs,student_features ,distilled_student_features = distil_model.forward(x_train,patches)
          _,preds = torch.max(outputs,1)

          teacher_features = teacher_model.forward_features(x_train)
          distilled_teacher_features = []
          for patch in patches:
             distilled_teacher_features.append(teacher_features[:,patch,:])

          distilled_teacher_features = torch.empty((teacher_features.shape[0],len(patches),teacher_features.shape[2]))
          for patch in patches:
            distilled_teacher_features = torch.index_select(teacher_features,1,patch)

          loss = distil_loss.forward(outputs,y_train,features_old=teacher_features,
                                     features_new=student_features,distilled_new_feats=distilled_student_features,
                                     distilled_old_feats=distilled_teacher_features)
          
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
        best_model_wts=copy.deepcopy(distil_model.state_dict())
        best_model = distil_model

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
  return distil_model.load_state_dict(best_model_wts)


model = train_model(distil_model=distil_model,teacher_model=teacher_model,optimizer=optimizer,scheduler=exp_lr_scheduler,num_epochs=epochs)
