import random
import os 
import glob
from dataset import MyDataset

def create_datasets():
  image_list = []

  file = open("/nas2/ckoutlis/DataStorage/vggface2/data/train_list.txt")

  for line in file:
    stripped_line = line.strip()
    stripped_line.replace("train","")
    image_list.append(stripped_line)

  random.shuffle(image_list)

  train_list = image_list[:round(len(image_list)*0.9)]
  eval_list = image_list[round(len(image_list)*0.9)+1:]

  dataset_used = "train/"

  for index in range (len(train_list)):
    train_list[index] = "/nas2/ckoutlis/DataStorage/vggface2/data/"+dataset_used+train_list[index]

  for index in range (len(eval_list)):
    eval_list[index] = "/nas2/ckoutlis/DataStorage/vggface2/data/"+dataset_used+eval_list[index]

  train_ds = MyDataset(train_list,True)
  eval_ds = MyDataset(eval_list,False)
  test_ds = []
  return train_ds, eval_ds,test_ds
