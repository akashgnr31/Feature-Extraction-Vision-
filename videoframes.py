import cv2
import glob

k=2
# Opens the Video file
for vid in glob.glob("/home/akash/Documents/Transpack/Datasets/Dy2video/*.3gp"):
    cap= cv2.VideoCapture(vid)
    # print(vid[50:58])
    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite("/home/akash/Documents/Transpack/Datasets/Dy2frame/"+vid[50:58]+"_"+str(i)+'.jpg',frame)
        i+=1
 
    cap.release()
    cv2.destroyAllWindows()