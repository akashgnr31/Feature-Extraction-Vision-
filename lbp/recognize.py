# import the necessary packages
from pyimagesearch.localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import SparsePCA
from sklearn.svm import SVC  
from sklearn.metrics import accuracy_score
import cv2
import glob
import numpy as np
 

def main():
    desc = LocalBinaryPatterns(10,8)
    data = []
    labels = []

    for n in glob.glob("/home/akash/Documents/Transpack/codes/lbp/training/*.png"):
        image = cv2.imread(n,0)
        hist = desc.describe(image)
        print('Training....')
        labels.append(n[51:58])
        data.append(hist)
        
# train a Linear SVM on the data
    transformer = SparsePCA(normalize_components=True)
    transformer.fit(data)
    X_transformed = transformer.transform(data) 
    # model = MultinomialNB(alpha=1)
    model = LinearSVC(C=100, random_state=32)
    model.fit(X_transformed, labels)
     

# loop over the testing images
    data_test=[]
    label_test=[]
    prediction=[]
    for n in glob.glob("/home/akash/Documents/Transpack/codes/lbp/test/*.png"):
        image = cv2.imread(n,0)
        hist = desc.describe(image)
        label_test.append(n[47:54])
        pr = model.predict(hist.reshape(1, -1))
        prediction.append(pr)
        print(pr)
        
    print(classification_report(label_test,prediction))   
    print("Accuracy-"+str(100*accuracy_score(label_test,prediction))) 

if __name__=="__main__":
    main()