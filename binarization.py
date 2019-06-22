import numpy as np 
import cv2
import glob
import matplotlib.pyplot as plt 

def main():
    image = []
    for img in glob.glob("/home/akash/Documents/Transpack/Datasets/ns_train/*.png"):
        n=cv2.imread(img,0)
        image.append(n)
        ret2,th2 = cv2.threshold(n,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cv2.imwrite("/home/akash/Documents/Transpack/Datasets/binarize/"+img[50:],th2)
        
    # for i in range(len(image)):
        # cv2.imshow('Otsu',th2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()




if __name__ == "__main__":
    main()