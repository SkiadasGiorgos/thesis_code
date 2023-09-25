import glob
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from tqdm import tqdm

text_path = "/fssd4/user-data/skiadasg/vggface2/data/test_list.txt"
df = pd.read_csv(text_path,header=None)

df[0] = '/fssd4/user-data/skiadasg/vggface2/data/test/' + df[0]

labels = os.listdir("/fssd4/user-data/skiadasg/vggface2/data/test/")
labels.sort()
label_list = []
count = 0
for idx in tqdm(df.index):
    count += 1
    for lab_idx in range(len(labels)):
        if labels[lab_idx] in df[0][idx]:
             label_list.append(lab_idx)

df[1] = label_list
train,test = train_test_split(df,test_size=0.1)

train_df = pd.DataFrame(train)
test_df = pd.DataFrame(test)

with open('/fssd4/user-data/skiadasg/vggface2/data/train.txt', 'w') as f:
    df_string = train_df.to_string(header=False, index=False)
    f.write(df_string)

with open('/fssd4/user-data/skiadasg/vggface2/data/test.txt', 'w') as f:
    df_string = test_df.to_string(header=False, index=False)
    f.write(df_string)