from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from vit_class import ViT
from tqdm.notebook import tqdm
import torch.optim as optim
from create_train_lists import create_datasets


model_checkpoint = 'google/vit-base-patch16-224-in21k'
seed = 42
lr = 1e-4
epochs = 5
batch_size = 32


config = {
    "model_ckp": model_checkpoint,
    "dataset": "VggFace2-test",
    "model": "BEiT",
    "seed": seed,
    "batch_size": batch_size,
    "lr": lr,
    "epochs": epochs   
}

train_data_path = "~/home/nas2/ckoutlis/DataStorage/vggface2/train"

train_ds, eval_ds, test_ds = create_datasets()  

train_dl = DataLoader(train_ds,batch_size=batch_size,shuffle=True,)
eval_dl = DataLoader(eval_ds,batch_size=batch_size,shuffle=True)
test_dl = DataLoader(test_ds,batch_size=batch_size,shuffle=True)

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


def model_train(dataloader, epochs, learning_rate):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Load nodel, loss function, and optimizer
    model = ViT().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Fine tuning loop
    for i in range(epochs):
        total_acc_train = 0
        total_loss_train = 0.0

        for train_image, train_label in tqdm(dataloader):
            output = model(train_image.to(device))
            loss = criterion(output, train_label.to(device))
            acc = (output.argmax(dim=1) == train_label.to(device)).sum().item()
            total_acc_train += acc
            total_loss_train += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f'Epochs: {i + 1} | Loss: {total_loss_train / dataset_size("train"): .3f} | Accuracy: {total_acc_train / dataset_size("train"): .3f}')

    return model

# Train the model
# trained_model = model_train(train_dl, epochs,lr)