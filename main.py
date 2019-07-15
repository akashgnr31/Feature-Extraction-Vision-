import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.transform import match_histograms
import feature_mapping as fm
import glob
import feature_mapping as fm
labels=[]
label=[]
for img in glob.glob("/home/akash/Documents/Transpack/Datasets/videoframes/cropped2/*.jpg"):
    labels.append(img[62:])
    label.append(img)
labels.sort()
label.sort()
labeling=labels[0]
k=0
for i in range(len(labels)):    
    if labeling[:4]!=labels[i][:4]:
        labeling=labels[i]
        if(k==i):
            break
        # print(label[k],label[i])
        img1=cv2.imread(label[k],1)
        img2=cv2.imread(label[i-1],1)
        # print(img1)
        # cv2.imshow("1",img1)
        # cv2.imshow("2",img2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print(label[k:i])
        fm.match(img1,img2,labels[k][:4])
        k=i
    


