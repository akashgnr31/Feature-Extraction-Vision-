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
path=[]
labels_full=[]
i=0
for img in glob.glob("/home/akash/Documents/Transpack/Datasets/training/*.jpg"):
    images.append(cv2.imread(img,0))
    labels.append(img[50:56])   #50-57
    print(img[50:56])
    path.append(img)
    labels_full.append(img[50:])
features_list=[]
for k in range(len(images)):
    f= open("training_feature_label.txt","a")
    horizontal_mask = np.array([-1, 0, 1])
    vertical_mask = np.array([[-1],[0],[1]])

    horizontal_gradient = HOG.calculate_gradient(images[k], horizontal_mask)
    vertical_gradient = HOG.calculate_gradient(images[k], vertical_mask)

    grad_magnitude = HOG.gradient_magnitude(horizontal_gradient, vertical_gradient)
    grad_direction = HOG.gradient_direction(horizontal_gradient, vertical_gradient)

    grad_direction = grad_direction % 180
    hist_bins = np.array([10,30,50,70,90,110,130,150,170])

# Histogram of the first cell in the first block.
    feature=[]
    for i in range(4,129,4):
        for j in range(4,65,4):
            cell_direction = grad_direction[i-4:i, j-4:j]
            cell_magnitude = grad_magnitude[i-4:i, j-4:j]
            HOG_cell_hist=HOG.HOG_cell_histogram(cell_direction, cell_magnitude, hist_bins)
            # print(HOG_cell_hist)
            feature=np.append(feature,HOG_cell_hist)
            # print(feature)
    # f.write(labels[k]+"\n")
    f.close()
    features_list.append(feature)
    print(np.shape(feature)) # 1152 x 1

# np.savetxt("training_feature.txt",features_list.astype(int))
hist_correct=[]
hist_incorrect=[]
correct=0
incorrect=0
total_correct=0
total_incorrect=0
for i in range(len(images)):
    for j in range(i+1,len(images)):
        simiarity=1-distance.cosine([features_list[i]], features_list[j])
        # print(simiarity)
        if labels[i]==labels[j]:
            hist_correct.append(simiarity)
            if simiarity>=0.66:
                correct+=1
            else:
                img1=cv2.imread(path[i],1)
                img2=cv2.imread(path[j],1)
                cv2.imshow("mismatch"+str(labels_full[i]),img1)
                cv2.imshow("mismatch2"+str(labels_full[i]),img2)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                print("Mismatch between"+labels_full[i]+" And  "+labels_full[j]+" with Similarity "+str(simiarity))
            total_correct+=1
        else :
            hist_incorrect.append(simiarity)
            if simiarity<0.66:
                
                incorrect+=1
            total_incorrect+=1
plt.figure(1)
plt.hist(hist_correct,1000,[0.4,1])
plt.title("same_test")
plt.figure(2)
plt.hist(hist_incorrect,1000,[0.4,1])
plt.title("different_test")
plt.show()
print("Correct Ones->"+str(correct)+"   "+"Missed_Ones->"+str(total_correct-correct))
print("correct_ones"+str(total_correct))
print("Accuracy_correct_ones->"+str((correct*100)/total_correct))
print("Incorrect_ones"+str(total_incorrect))
print("Correct Ones->"+str(incorrect)+"   "+"Missed_Ones->"+str(total_incorrect-incorrect))
print("Accuracy_incorrect_ones->"+str((incorrect*100)/total_incorrect))
