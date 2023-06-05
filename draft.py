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

model_checkpoint = 'google/vit-base-patch16-224-in21k'
seed = 42
lr = 1e-4
epochs = 5
batch_size = 256
save_path = "/home/skiadasg/thesis_code/results/results.pkl"

config = {
    "model_ckp": model_checkpoint,
    "dataset": "VggFace2-test",
    "seed": seed,
    "batch_size": batch_size,
    "lr": lr,
    "epochs": epochs   
}

train_data_path = "~/home/nas2/ckoutlis/DataStorage/vggface2/train"

train_ds, eval_ds, test_ds = create_datasets()  

train_dl = DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=32)
eval_dl = DataLoader(eval_ds,batch_size=batch_size,shuffle=True,num_workers=32)
test_dl = DataLoader(test_ds,batch_size=batch_size,shuffle=True,num_workers=32)

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
# device = torch.device('cpu')

# model =  timm.create_model("vit_base_patch16_clip_384.openai_ft_in12k_in1k",pretrained = True)
model =  timm.create_model("vit_base_patch16_clip_224.openai_ft_in12k",pretrained = True)

for param in model.parameters():
  param.requires_grad =  False
for param in model.blocks[11:]:
  param.requires_grad = True
  
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
optimizer = optim.Adam(model.head.parameters(), lr=lr,  weight_decay=0.0001)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=3, gamma=0.97)

results = []

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
        pass
        scheduler.step()


      epoch_loss = running_loss/dataset_size(phase)
      epoch_acc = running_corrects.double()/dataset_size(phase)

        
      print("Loss: {:.4f} Acc: {:.4f}".format(epoch_loss, epoch_acc))


      if  phase == "val"  and epoch_acc>best_acc:
        best_model_wts=copy.deepcopy(model.state_dict())
        results.append({"config": config, "metrics": metrics})
      
      with open(save_path, "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

      
      if phase == "train":
        metrics["train_loss"].append(epoch_loss)
        metrics["train_acc"].append(epoch_acc)
      elif phase == "val":
        metrics["val_loss"].append(epoch_loss)
        metrics["val_acc"].append(epoch_acc)

  return model.load_state_dict(best_model_wts)

def test_model():
  test_loss = 0.0
  running_loss  = 0.0
  running_corrects = 0.0

  for x,y in tqdm(dataloaders(phase="test")):
    data = x.to(device) 
    target = y.to(device)
    
    with torch.no_grad(): # turn off autograd for faster testing
      output = model(data)
      loss = criterion(output, target)

    _, pred = torch.max(output, 1)
    running_loss += loss.item()*data.size(0)
    running_corrects += torch.sum(pred == target.data)  

  test_loss = running_loss/dataset_size("test")
  test_acc = running_corrects.double()/dataset_size("test")
  
  print("Loss: {:.4f} Acc: {:.4f}".format(test_loss, test_acc))


# model = train_model(model=model,optimizer=optimizer,scheduler=exp_lr_scheduler,num_epochs=epochs)
test_model()