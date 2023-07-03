from distillation import train_model, distillation_process
import torch
import os

num_epochs = 4
model_save_path = "/home/skiadasg/thesis_code/thesis_code/results/test_model.pkl"
total_classes = len(os.listdir("/nas2/ckoutlis/DataStorage/vggface2/data/test/"))
new_classes_number = 400
class_increment = 20

train_ds,val_ds,train_dl,val_dl,distil_model,teacher_model = distillation_process(old_classes_number=0,new_classes_number=new_classes_number,
                                                                   model_ckp='vit_small_r26_s32_224.augreg_in21k_ft_in1k')

model = train_model(distil_model=distil_model,teacher_model=teacher_model,num_epochs=num_epochs,
                            train_dl=train_dl,eval_dl=val_dl,train_ds=train_ds, eval_ds=val_ds)

old_classes_number = new_classes_number
new_classes_number = old_classes_number+class_increment

while total_classes>=old_classes_number+new_classes_number:
    
    print("old classes: "+old_classes_number+"\n"+"new classes: "+new_classes_number+"\n")
    model_ckp = model_save_path
    train_ds,val_ds,train_dl,val_dl,distil_model,teacher_model = distillation_process(old_classes_number=old_classes_number
                                                                                      ,new_classes_number=new_classes_number,
                                                                    model_ckp=model_ckp)

    model = train_model(distil_model=distil_model,teacher_model=teacher_model,num_epochs=num_epochs,
                                train_dl=train_dl,eval_dl=val_dl,train_ds=train_ds, eval_ds=val_ds)

    old_classes_number = new_classes_number
    new_classes_number = old_classes_number+class_increment

