import cv2
import glob
import matplotlib.pyplot as pyplot
import numpy as np
def main():
    images=[]
    labels=[]
    for img in glob.glob("/home/akash/Documents/Transpack/Datasets/videoframes/cropped/*.jpg"):
        labels.append(img[61:])
    labels.sort()
    for i in range(len(labels)):
        n=cv2.imread("/home/akash/Documents/Transpack/Datasets/videoframes/cropped/"+labels[i],0)
        images.append(n)
    result=[]
    suming=images[0]
    labeling=labels[0]
    for i in range(len(images)):    
        if labeling[:8]==labels[i][:8]:
            print(i)
            print(labels[i])
            suming=np.minimum(suming,images[i])
        else:
            result.append(suming)
            cv2.imwrite("/home/akash/Documents/Transpack/Datasets/videoframes/cropped/"+labeling[:8]+str("av.jpg"),suming)
            suming=images[i]
            labeling=labels[i]
        
    
    # cv2.imshow("average",result)
    # cv2.imwrite("/home/akash/Documents/Transpack/Datasets/Cropped/4/onlytags/average.jpg",result)
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()