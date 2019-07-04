import cv2
import numpy as np
from imageio import imread
import glob

images=[]
labels=[]
def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


for img in glob.glob("/home/akash/Documents/Transpack/Datasets/videoframes/cropped/*.jpg"):
    labels.append(img[61:])
labels.sort()
for i in range(len(labels)):
    n=cv2.imread("/home/akash/Documents/Transpack/Datasets/videoframes/cropped/"+labels[i],0)
    images.append(n)
result=[]
for i in range(0,len(images)-1,2):
    print(labels[i][:4])
    suming=images[i]
    news=images[i+1]
    flow = cv2.calcOpticalFlowFarneback(news, suming, None, 0.5, 9, 30, 3, 5, 1.2, 0)
    hsv = draw_hsv(flow)
    news_war = warp_flow(news, flow)
    print(news_war)
    cv2.imshow("I",suming)
    cv2.imshow("I+1",news)
    cv2.imshow("warped",news_war)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
