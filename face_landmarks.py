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

save_path = "landmarks.pkl"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(margin=20,keep_all=True,post_process=False,device=device)
image_list = glob.glob("/nas2/ckoutlis/DataStorage/vggface2/data/test/*/*.jpg")

frame = Image.open('/nas2/ckoutlis/DataStorage/vggface2/data/test/n006497/0029_02.jpg')
frame = frame.resize([224,224])
_,_,landmarks = mtcnn.detect(frame,landmarks=True)


anchor_points = []
for i in range(0,224,32):
    for j in range(0,224,32):
        anchor_points.append([i,j])

patches,center_patches = calculate_landmark_patches(landmarks)
# print(patches)
# print(landmarks)
# print(patch_centers)
# Landmarks in random photo
fig,ax = plt.subplots()
ax.imshow(frame)
ax.axis('off')
iter =0
for anchor in anchor_points:    
    rectangle = Rectangle(anchor,32,32,linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rectangle)
for  landmark in landmarks:
    ax.scatter(landmark[:,0],landmark[:,1],s=8)
for center in center_patches:
    ax.scatter(center[0],center[1],s=8,c='w')

fig.savefig("/home/skiadasg/thesis_code/thesis_code/rand.png")

# landmark_list = []
# c =0 
# for image in image_list:
#     frame = Image.open(image)
#     frame = frame.resize([224,224])
#     _,_,landmarks = mtcnn.detect(frame,landmarks=True)
#     landmark_patch = calculate_landmark_patches(landmarks)
#     temp_dict = {"image":image,
#                  "landmarks":landmark_patch}
#     with open(save_path,'ab') as h:
#         pickle.dump(temp_dict,h)
#     c+=1
#     if c==10:
#         break 


# objects = []
# with (open(save_path, "rb")) as openfile:
#     while True:
#         try:
#             objects.append(pickle.load(openfile))
#             # objects.append(torch.load(save_path,map_location=torch.device('cpu')))
#         except EOFError:
#             break

# print(objects)