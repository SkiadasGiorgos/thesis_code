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
# model_checkpoint = 'resnet50.a1_in1k'

seed = 42
lr = 1e-4
epochs = 10
batch_size = 32
save_path = "/home/skiadasg/thesis_code/results/results.pkl"
model_save_path = "/home/skiadasg/thesis_code/results/resnet_model_vgg.pkl"

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

model = timm.create_model(model_checkpoint,pretrained=True,num_classes=len(os.listdir("/nas2/ckoutlis/DataStorage/vggface2/data/train")))

'''
  When using any model that does not contain HEAD
'''
# model.fc = nn.Sequential(nn.Linear(in_features=2048,out_features=512,bias=False),
#                          nn.ReLU(),
#                          nn.BatchNorm1d(512),
#                          nn.Dropout(.4),
#                          nn.Linear(in_features=512,out_features=len(os.listdir("/nas2/ckoutlis/DataStorage/vggface2/data/test")),bias=False))

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

debarcle_layers(model,3g0)

# Replacing model's last layer in order to adjust the final results to our dataset.

n_inputs = model.head.in_features
model.head = nn.Sequential(
    nn.Linear(n_inputs,512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512,len(os.listdir("/nas2/ckoutlis/DataStorage/vggface2/data/train"))) #!!!Output size must match number of labels!!!!
)

model.to(device)

criterion = LabelSmoothingCrossEntropy()
criterion=criterion.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr,)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=3, gamma=0.97)


results = []
results.append(config)

def train_model(model,optimizer,scheduler,num_epochs):
  print(device)
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
          outputs  = model(x_train)
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

    results.append(metrics)
      
    with open(save_path, "wb") as h:
      pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

  
  torch.save(best_model,model_save_path)
  return model.load_state_dict(best_model_wts)


model = train_model(model=model,optimizer=optimizer,scheduler=exp_lr_scheduler,num_epochs=epochs)

