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

for j in y:
    for i in x:
        centroids.append([i+square_centroid_dist,j-square_centroid_dist])

patch_centers = []
for index,value in enumerate(centroids):
    patch_centers.append([index+1,value])

def calculate_landmark_patches(landmarks):
    landmark_patches = []
    center_patches = []
    for j in range(5): # 5 is the number of landmarks found by mtcnn
        min_dist = np.inf
        # for i in range(len(patch_centers)):
        #     diff = landmarks[0][j] - np.array(patch_centers[i][1])
        #     dot_prod = np.dot(diff.T,diff)
        #     dist = np.sqrt(dot_prod)
        #     if dist < min_dist:
        #         patch_temp = patch_centers[i][0]
        #         min_dist = dist
        for i in range(len(patch_centers)):
            dist = np.linalg.norm(landmarks[0][j]-patch_centers[i][1])
            if dist < min_dist:
                patch_temp = patch_centers[i][0]
                centroid_temp = patch_centers[i][1]
                min_dist = dist
        landmark_patches.append(patch_temp)
        center_patches.append(centroid_temp)
    return landmark_patches, center_patches
