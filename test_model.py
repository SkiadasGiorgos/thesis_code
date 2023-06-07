from tqdm import tqdm
import torch

def test_model(model,device,test_dl,criterion,dataset_size):
  test_loss = 0.0
  running_loss  = 0.0
  running_corrects = 0.0

  for x,y in tqdm(test_dl):
    data = x.to(device) 
    target = y.to(device)
    
    with torch.no_grad(): # turn off autograd for faster testing
      output = model(data)
      loss = criterion(output, target)

    _, pred = torch.max(output, 1)
    running_loss += loss.item()*data.size(0)
    running_corrects += torch.sum(pred == target.data)  

  test_loss = running_loss/dataset_size
  test_acc = running_corrects.double()/dataset_size
  
  print("Loss: {:.4f} Acc: {:.4f}".format(test_loss, test_acc))
