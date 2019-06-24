import skimage.io, skimage.color
import numpy as np
import matplotlib.pyplot
import HOG
import glob
import cv2
from scipy.spatial import distance
import matplotlib.pyplot as plt

images=[]
labels=[]
labels_full=[]
i=0
for img in glob.glob("/home/akash/Documents/Transpack/Datasets/training/*.png"):
    images.append(cv2.imread(img,0))
    labels.append(img[50:57])
features_list=[]
    
for k in range(len(images)):
    horizontal_mask = np.array([-1, 0, 1])
    vertical_mask = np.array([[-1],[0],[1]])

    horizontal_gradient = HOG.calculate_gradient(images[k], horizontal_mask)
    vertical_gradient = HOG.calculate_gradient(images[k], vertical_mask)

    grad_magnitude = HOG.gradient_magnitude(horizontal_gradient, vertical_gradient)
    grad_direction = HOG.gradient_direction(horizontal_gradient, vertical_gradient)

    grad_direction = grad_direction % 180
    hist_bins = np.array([10,30,50,70,90,110,130,150,170])

    feature=[]
    for i in range(8,129,8):
        for j in range(8,65,8):
            cell_direction = grad_direction[i-8:i, j-8:j]
            cell_magnitude = grad_magnitude[i-8:i, j-8:j]
            HOG_cell_hist=HOG.HOG_cell_histogram(cell_direction, cell_magnitude, hist_bins)
            # print(HOG_cell_hist)
            feature=np.append(feature,HOG_cell_hist)
            # print(feature)
            
    features_list.append(feature.astype(int))
print(np.shape(feature)) # 288 x 1
same_eucledian=[]
different_eucledian=[]
correct=0
incorrect=0
total_correct=0
total_incorrect=0
for i in range(len(images)):
    for j in range(i+1,len(images)):
        simiarity=1-distance.cosine([features_list[i]], features_list[j])
        if labels[i]==labels[j]:
            same_eucledian.append(distance.euclidean(features_list[i], features_list[j]))
            if simiarity>=0.50:
                correct+=1
            total_correct+=1
        else :
            different_eucledian.append(distance.euclidean(features_list[i], features_list[j]))
            if simiarity<0.50:
                incorrect+=1
            total_incorrect+=1
plt.figure(1)
plt.hist(same_eucledian,1000,[100000,500000])
plt.title("same")
plt.figure(2)
plt.hist(different_eucledian,1000,[100000,500000])
plt.title("different")
plt.show()

print("Correct Ones->"+str(correct)+"   "+"Missed_Ones->"+str(total_correct-correct))
print("correct_ones"+str(total_correct))
print("Accuracy_correct_ones->"+str((correct*100)/total_correct))
print("Correct Ones->"+str(incorrect)+"   "+"Missed_Ones->"+str(total_incorrect-incorrect))
print("Incorrect_ones"+str(total_incorrect))
print("Accuracy_incorrect_ones->"+str((incorrect*100)/total_incorrect))
