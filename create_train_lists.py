import random
import os 
import glob
from dataset import MyDataset
import numpy as np

def create_datasets():
  image_list = []

  file = open("/nas2/ckoutlis/DataStorage/vggface2/data/test_list.txt")
  dataset_used = "test/"

  for line in file:
    stripped_line = line.strip()
    stripped_line.replace("train","")
    image_list.append(stripped_line)

  random.shuffle(image_list)

  train_list = image_list[:round(len(image_list)*0.9)]
  eval_list = image_list[round(len(image_list)*0.9)+1:]


  for index in range (len(train_list)):
    train_list[index] = "/nas2/ckoutlis/DataStorage/vggface2/data/"+dataset_used+train_list[index]

  for index in range (len(eval_list)):
    eval_list[index] = "/nas2/ckoutlis/DataStorage/vggface2/data/"+dataset_used+eval_list[index]

  train_ds = MyDataset(train_list,True)
  eval_ds = MyDataset(eval_list,False)
  test_ds = []
  return train_ds, eval_ds,test_ds

"""
Arguments:
  old classes: the number of old classes we would like to use
  new classes: the number of new classes we would like to use
  percentage_of_old_samples: the percentage of old samples that will be used for the distillation process

The function finds all the available classes va glob, then sorts them and chooses the expected number.
Sorting the function helps with recursively creating datasets without class collision. Then the images
belonging to each class are stored and in image lists, the temp image list is shuffled and the  correct 
amount of old samples is then used. Finally the datasets are created.  
"""

def create_distil_dataset(old_classes_number=450,new_classes_number=10,percentage_of_old_samples=0.3):
  
  image_list = []

  file_path = "/nas2/ckoutlis/DataStorage/vggface2/data/test/*"
  file = open("/nas2/ckoutlis/DataStorage/vggface2/data/test_list.txt")
  dataset_used = "test/"


  class_list = sorted(glob.glob(file_path))
  for i in range(len(class_list)):
    class_list[i] = class_list[i].split('/')[-1]

  old_classes = class_list[:old_classes_number]
  new_classes = class_list[old_classes_number:old_classes_number+new_classes_number]

  
  image_list_temp = []

  for line in file:
    stripped_line = line.strip()
    for substring in new_classes:
      if substring in line:
        image_list.append(stripped_line)
    for substring in old_classes:
      if substring in line:
        image_list_temp.append(stripped_line)

  random.shuffle(image_list_temp)


  # Old classes selection
  if old_classes_number != 0:
    image_list_temp = image_list_temp[:int(np.floor(percentage_of_old_samples*len(image_list_temp)))]
  
  image_list = np.concatenate((image_list,image_list_temp))

  
  random.shuffle(image_list)


  train_list = image_list[:round(len(image_list)*0.9)]
  eval_list = image_list[round(len(image_list)*0.9)+1:]
  
  train_paths = []
  val_paths = []


  for index in range (len(train_list)):
    train_paths.append("/nas2/ckoutlis/DataStorage/vggface2/data/"+dataset_used+train_list[index])
    
  
  for index in range (len(eval_list)):
    val_paths.append("/nas2/ckoutlis/DataStorage/vggface2/data/"+dataset_used+eval_list[index]) 

  train_ds = MyDataset(train_paths,True)
  eval_ds = MyDataset(val_paths,False)

  return train_ds,eval_ds
