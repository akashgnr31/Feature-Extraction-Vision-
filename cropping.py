import numpy as np 
import glob
import cv2
def main():
    i=1
    for n in glob.glob("/home/akash/Documents/Transpack/Datasets/*.jpg"):
        img = cv2.imread(n,0)
        print(i)
        # crop_img = img[84:320, 101:712]
        scale_percent = 60 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
# resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        cv2.imwrite("/home/akash/Documents/Transpack/Datasets/"+n[41:],resized)
        i=i+1


if __name__ == "__main__":
    main()