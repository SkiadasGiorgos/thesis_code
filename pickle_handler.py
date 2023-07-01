import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np

save_path = "/home/skiadasg/thesis_code/thesis_code/results/results.pkl"

torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

objects = []
with (open(save_path, "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
            # objects.append(torch.load(save_path,map_location=torch.device('cpu')))
        except EOFError:
            break

config = objects[0][0]
results = objects[0][-1]

for figure in ["acc","loss"]:
    train_str = "train_"+figure
    val_str = "val_"+figure  
    title = config["model_ckp"]+"_"+figure

    train = results[train_str]
    val = results[val_str]
    file_name = config["model_ckp"]+figure
    save_path = "/home/skiadasg/thesis_code/thesis_code/plots/"+title+".png"

    plt.figure()
    plt.plot(train)
    plt.plot(val)
    plt.legend([train_str,val_str])
    plt.title(title)
    plt.savefig(save_path)