import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt
from skimage.transform import match_histograms
def imagefilering(tempImg):
	blur = cv2.GaussianBlur(tempImg,(3,3),0)
	smooth= cv2.addWeighted(blur,1.5,tempImg,-0.5,0)
	return smooth
def cropping(IMG1,IMG2):
	# Create our mask by selecting the non-zero values of the picture
	ret, mask = cv2.threshold(IMG2,0,255,cv2.THRESH_BINARY)

# Select the contour
	mask , cont, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
# if your mask is incurved or if you want better results, 
# you may want to use cv2.CHAIN_APPROX_NONE instead of cv2.CHAIN_APPROX_SIMPLE, 
# but the rectangle search will be longer

	cv2.drawContours(IMG2, cont, -1, (255,0,0), 1)
	# cv2.imshow("Your picture with contour", IMG2)
	# cv2.waitKey(0)

# Get all the points of the contour
	contour = cont[0].reshape(len(cont[0]),2)

# we assume a rectangle with at least two points on the contour gives a 'good enough' result
# get all possible rectangles based on this hypothesis
	rect = []

	for i in range(len(contour)):
		x1,y1=contour[i]
	    
		for j in range(len(contour)):
			x2,y2 = contour[j]
			area = ((abs(y2-y1))*(abs(x2-x1)))
			rect.append(((x1,y1), (x2,y2), area))

# the first rect of all_rect has the biggest area, so it's the best solution if he fits in the picture
	all_rect = sorted(rect, key = lambda x : x[2], reverse = True)

# we take the largest rectangle we've got, based on the value of the rectangle area
# only if the border of the rectangle is not in the black part

# if the list is not empty
	if all_rect:

		best_rect_found = False
		index_rect = 0
		nb_rect = len(all_rect)

    # we check if the rectangle is  a good solution
		while not best_rect_found and index_rect < nb_rect:

			rect = all_rect[index_rect]
			(x1, y1) = rect[0]
			(x2, y2) = rect[1]

			valid_rect = True

        # we search a black area in the perimeter of the rectangle (vertical borders)
			x = min(x1, x2)
			while x <max(x1,x2)+1 and valid_rect:
				if mask[y1,x] == 0 or mask[y2,x] == 0:
                # if we find a black pixel, that means a part of the rectangle is black
                # so we don't keep this rectangle
					valid_rect = False
				x+=1

			y = min(y1, y2)
			while y <max(y1,y2)+1 and valid_rect:
				if mask[y,x1] == 0 or mask[y,x2] == 0:
				    	valid_rect = False
				y+=1

			if valid_rect:
				best_rect_found = True

			index_rect+=1

		if best_rect_found:

			# cv2.rectangle(IMG2, (x1,y1), (x2,y2), (255,0,0), 1)
			# cv2.imshow("Is that rectangle ok?",IMG2)
			# cv2.waitKey(0)

        # Finally, we crop the picture and store it
			print(x1,x2,y1,y2)
			IMG1 = IMG1[min(y1, y2):max(y1, y2), min(x1,x2):max(x1,x2)]
			IMG2 = IMG2[min(y1, y2):max(y1, y2), min(x1,x2):max(x1,x2)]
		else:
			print("No rectangle was detected")
			exit()
	return IMG1,IMG2

def registratiom(Img1,Img2):
	feature = cv2.xfeatures2d.SIFT_create()

	(kps, des) = feature.detectAndCompute(Img1, None)
	(kps2, des2) = feature.detectAndCompute(Img2, None)


	print("A #  {} keypoints".format(len(kps)))
	print("B #  {} keypoints".format(len(kps2)))

	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	# search_params = dict(checks=50)   # or pass empty dictionary
	# bf = cv2.BFMatcher()
	# matches = bf.knnMatch(des,des2, k=2)


	flann = cv2.FlannBasedMatcher(index_params,{})
	matches = flann.knnMatch(des,des2,k=2)
	matchesMask = [[0,0] for i in range(len(matches))]   # drawing good matches

	good = []
	for m,n in matches:
		if m.distance < 0.6*n.distance:
			good.append(m)
    
	MIN_MATCH_COUNT = 15
	print(len(good))

	if len(good)>=MIN_MATCH_COUNT:
		src_pts = np.float32([ kps[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kps2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
		f = open('Mapping/points_img1.txt','w')
		f2 = open('Mapping/points_img2.txt','w')
		for x in range(src_pts.size):
			if x % 2 == 0:
				f.write(str(src_pts.item(x))+" ")
				f2.write(str(dst_pts.item(x))+" ")
			else:
				f.write(str(src_pts.item(x))+"\n")
				f2.write(str(dst_pts.item(x))+"\n")
		f.close()
		f2.close()
		print("Files generated")
		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
		print("Homography calculated via in-built function : ")
		print(M)

		matchesMask = mask.ravel().tolist()
		h,w = Img1.shape[0],Img1.shape[1]
		Output = cv2.warpPerspective(Img1, M, (len(Img2[1]),len(Img2)))
		return Output
	else:
		print("Not enough good matches")
		exit()

def match(img,img2,labels):
	img2 = imagefilering(img2)
	img= imagefilering(img)
	img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
	img2 = cv2.equalizeHist(img2)
	im_out=registratiom(img,img2)
	im_out=cv2.cvtColor(im_out,cv2.COLOR_BGR2GRAY)
	im_out = cv2.equalizeHist(im_out)
	# cv2.imshow("real",img)
	# cv2.imshow("transformed",im_out)
	# cv2.waitKey(0)
	img2,im_out=cropping(img2,im_out)
	# cv2.imshow("output1",img2)
	# cv2.imshow("output2",im_out)
	result_obtained=np.minimum(img2,im_out)

	cv2.imwrite("/home/akash/Documents/Transpack/codes/Homography/Images/"+labels+"new_av.jpg",result_obtained)
	

	




# match()
	# img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
	# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
	#     singlePointColor = None,
	#     matchesMask = matchesMask, # draw only inliers
	#     flags = 2)
	# img3 = cv2.drawMatches(img,kps,img2,kps2,good,None,**draw_params)
	# plt.imshow(img3, 'gray'),plt.show()
