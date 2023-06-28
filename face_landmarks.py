from facenet_pytorch import MTCNN
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from transforms import transforms
import glob
import pickle
from patches_dims import calculate_landmark_patches, centroids,patch_centers
from matplotlib.patches import Rectangle
from tqdm import tqdm

save_path = "landmarks_new.pkl"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(margin=20,keep_all=True,post_process=False,device=device)
image_list = glob.glob("/nas2/ckoutlis/DataStorage/vggface2/data/test/*/*.jpg")

all_landmarks = []
all_paths = []
"""
    Using mtcnn to detect face landmarks and calculating patch centers closer to them
    , in some cases mtcnn returns None, if thats the case the previous landmark is used.  
"""
nones = 0
dictionary = {}
count = 0
for image in tqdm(image_list):
    count+=1
    if count ==10: 
        break
    frame = Image.open(image).convert("RGB")
    frame = frame.resize([224,224])
    _,_,landmarks = mtcnn.detect(frame,landmarks=True)
    if landmarks is None:
        fig,ax = plt.subplots()
        ax.imshow(frame)
        fig.savefig("/home/skiadasg/thesis_code/thesis_code/rand.png")
        nones += 1
        continue
    landmark_patch = calculate_landmark_patches(landmarks)
    all_landmarks.append(landmark_patch)
    all_paths.append(image)
    temp_dict = {image:landmark_patch,
                }
    dictionary.update(temp_dict)

with open(save_path,'ab') as h:
    pickle.dump(dictionary,h)

print(nones)
print(len(image_list))

# Plotting landmarks, patches and selected centers in random photo

# frame = Image.open('/nas2/ckoutlis/DataStorage/vggface2/data/test/n006497/0029_02.jpg')
# frame = frame.resize([224,224])
# _,_,landmarks = mtcnn.detect(frame,landmarks=True)


# anchor_points = []
# for i in range(0,224,32):
#     for j in range(0,224,32):
#         anchor_points.append([i,j])

# patches,center_patches = calculate_landmark_patches(landmarks)
# print(patches)

# # Landmarks in random photo
# fig,ax = plt.subplots()
# ax.imshow(frame)
# ax.axis('off')
# iter =0
# for anchor in anchor_points:    
#     rectangle = Rectangle(anchor,32,32,linewidth=1, edgecolor='r', facecolor='none')
#     ax.add_patch(rectangle)
# for  landmark in landmarks:
#     ax.scatter(landmark[:,0],landmark[:,1],s=8)
# for center in center_patches:
#     ax.scatter(center[0],center[1],s=8,c='w')

# fig.savefig("/home/skiadasg/thesis_code/thesis_code/rand.png")
