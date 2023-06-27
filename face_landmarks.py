from facenet_pytorch import MTCNN
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from transforms import transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(margin=20,keep_all=True,post_process=False,device=device)

frame = Image.open("/nas2/ckoutlis/DataStorage/vggface2/data/test/n008655/0558_01.jpg")
frame.resize([224,224])
boxes,probs,landmarks = mtcnn.detect(frame,landmarks=True)

print(landmarks)


# Landmarks in random photo
# fig,ax = plt.subplots()
# ax.imshow(frame)
# ax.axis('off')

# for  landmark in landmarks:
#     ax.scatter(landmark[:,0],landmark[:,1],s=8)
# fig.savefig("/home/skiadasg/thesis_code/thesis_code/rand.png")