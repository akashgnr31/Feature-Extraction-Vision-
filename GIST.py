import numpy as np
from skimage import transform
import cv2
import skimage.io, skimage.color
import numpy as np
import HOG
import glob
from scipy.spatial import distance
import matplotlib.pyplot as plt
import gist
images=[]
labels=[]
path=[]
labels_full=[]
features_list=[]
i=0
for imagepath in glob.glob("/home/akash/Documents/Transpack/Datasets/training/*.png"):
    imsize = (171, 131)
    features = []
    img=cv2.imread(imagepath)
    images.append(img)
    labels.append(imagepath[50:57])
    path.append(imagepath)
    labels_full.append(imagepath[50:])
    print(i)
    i+=1
    img_resized = transform.resize(img, imsize, preserve_range=True).astype(np.uint8)
    feature = gist.extract(img_resized)
    features_list.append(feature)
    
    
hist_correct=[]
hist_incorrect=[]
correct=0
incorrect=0
total_correct=0
total_incorrect=0
for i in range(len(images)):
    for j in range(i+1,len(images)):
        print(np.shape(features_list[i]))
        simiarity=1-distance.cosine(features_list[i], features_list[j])
        # print(simiarity)
        if labels[i]==labels[j]:
            hist_correct.append(simiarity)
            if simiarity>=0.975:
                correct+=1
            else:
                img1=cv2.imread(path[i],1)
                img2=cv2.imread(path[j],1)
                # cv2.imshow("mismatch"+str(labels_full[i]),img1)
                # cv2.imshow("mismatch2"+str(labels_full[i]),img2)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                print("Mismatch between"+labels_full[i]+" And  "+labels_full[j]+" with Similarity "+str(simiarity))
            total_correct+=1
        else :
            hist_incorrect.append(simiarity)
            if simiarity<0.975:
                
                incorrect+=1
            total_incorrect+=1
plt.figure(1)
plt.hist(hist_correct,1000,[0.9,1])
plt.title("same_test")
plt.figure(2)
plt.hist(hist_incorrect,1000,[0.9,1])
plt.title("different_test")
plt.show()
print("Correct Ones->"+str(correct)+"   "+"Missed_Ones->"+str(total_correct-correct))
print("correct_ones"+str(total_correct))
print("Accuracy_correct_ones->"+str((correct*100)/total_correct))
print("Incorrect_ones"+str(total_incorrect))
print("Correct Ones->"+str(incorrect)+"   "+"Missed_Ones->"+str(total_incorrect-incorrect))
print("Accuracy_incorrect_ones->"+str((incorrect*100)/total_incorrect))