import os
from os import listdir
from PIL import Image
from tqdm import tqdm
import glob

count=0
for filename in tqdm(glob.glob('/fssd4/user-data/skiadasg/vggface2/data/train/*/*')):
    if filename.endswith('.jpg'):
     try:
      img=Image.open(filename)
      img.verify()
     except(IOError,SyntaxError)as e:
         print('Bad file  :  '+filename)
         count=count+1
         print(count)