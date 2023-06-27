from PIL import Image
from torch import nn
import numpy as np

image_height = 224
image_width = 224
patch_size = 32
square_centroid_dist = patch_size//2

y = np.arange(patch_size,image_height+patch_size,patch_size)
y = np.flip(y)
x = np.arange(0,image_width,patch_size)
centroids = []

for i in x:
    for j in y:
        centroids.append([i+square_centroid_dist,j-square_centroid_dist])

patch_centers = []
for index,value in enumerate(centroids):
    patch_centers.append([index,value])

min_dist = np.inf
landmarks = [[126.48541259765625, 124.46916198730469],
  [179.53973388671875, 126.32745361328125],
  [161.130126953125, 152.90260314941406],
  [131.11929321289062, 186.91676330566406],
  [176.72128295898438, 187.10568237304688]]

for landmark in landmarks:
    for i in len(patch_centers):
        dist = np.linalg.norm(landmark,patch_centers[i,:])