import os
import shutil
import numpy as np
from PIL import Image
from skimage import data
from sklearn.cluster import KMeans

for path in os.listdir('./images'):
    img = Image.open('./images/'+path)
    img = img.convert('RGB')
    img_resize = img.resize((100, 100))
    img_resize.save('./convert/'+path)

feature = np.array([data.imread('./convert/'+path) for path in os.listdir('./convert')])
feature = feature.reshape(len(feature), -1).astype(np.float64)

model = KMeans(n_clusters=2).fit(feature)

labels = model.labels_

for label, path in zip(labels, os.listdir('./convert')):
    print("label:"+str(label)+",path:"+str(path))
    try:
        os.makedirs("./group/"+str(label))
    except:
        pass
    shutil.copyfile('./images/'+path, './group/'+str(label)+'/'+path)
    print(label, path)
