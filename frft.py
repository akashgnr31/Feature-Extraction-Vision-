import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

def main():
    images = []
    for img in glob.glob("/home/akash/Documents/Transpack/codes/frft/dataset/*.jpg"):
        n= cv2.imread(img,0)
        images.append(n)
    cosine=[]
    sine=[]
    for i in range(len(images)):
        f = np.fft.fft2(images[i])
        fshift = np.fft.fftshift(f)
        fphase=(fshift)/(np.abs(fshift))
        cos=fphase.real
        sin=fphase.imag
        cosine.append(cos)
        sine.append(sin)
        # plt.figure()
        # magnitude_spectrum = 20*np.log(np.abs(fshift))
        # plt.subplot(1,1,1)    
        # print("aksh")
        # plt.imshow(magnitude_spectrum)
        # plt.savefig('/home/akash/Documents/Transpack/codes/frft/'+str(i+1)+'.jpg')
    # for i in range(len(cosine)):
    #     for j in range(i+1,len(cosine)):
    #         print(np.linalg.norm(sine[i]-sine[j]))
    
    print(np.shape(sine[2]))
    # print(distance.euclidean(sine[0],sine[1]))
    print(np.linalg.norm(sine[1]-sine[3]))
    print(np.linalg.norm(sine[3]-sine[0]))
    # cv2.imshow('0',images[0])
    # cv2.imshow('1',images[1])
    # cv2.imshow('2',images[3]) 
    cv2.waitKey()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()