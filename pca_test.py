import os
import shutil
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def img2pixarray(filename):
    # on HSV color space
    img = cv2.imread(filename) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # shrink image (640x640->64x64)
    img = cv2.resize(img, (64,64))
    # transform images into vectors with shape (1, 64*64*3)
    img = np.reshape(img, (1, img.shape[0]*img.shape[1]*img.shape[2]))
    return img

# import images
image_path = "images/*.jp*g"
path_list = glob.glob(image_path)

pca = PCA(n_components=1)
for index, path in enumerate(path_list):
    img_pix = img2pixarray(path)
    if index is 0:
        pix_list = img_pix
    else:
        pix_list = np.append(pix_list, img_pix, axis=0)
else:
    pcaed = pca.fit_transform(pix_list)

import pdb;pdb.set_trace()

# create a list of the 1st principal component and file names
image_list = []
for index, path in enumerate(path_list):
    image_list.append((pcaed[index, 0], path))

# sort list by parameters
sorted_list = sorted(image_list, key=lambda tgt: tgt[0], reverse=True) 

result_dir = "results"
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
for i, info in enumerate(sorted_list):
    new_path = os.path.join(result_dir, "{0:08d}.jpg".format(i))
    temp_path = info[1]
    shutil.copyfile(temp_path, new_path)
