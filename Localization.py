import cv2
import numpy as np

"""
In this file, you need to define plate_detection function.
To do:
	1. Localize the plates and crop the plates
	2. Adjust the cropped plate images
Inputs:(One)
	1. image: captured frame in CaptureFrame_Process.CaptureFrame_Process function
	type: Numpy array (imread by OpenCV package)
Outputs:(One)
	1. plate_imgs: cropped and adjusted plate images
	type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
Hints:
	1. You may need to define other functions, such as crop and adjust function
	2. You may need to define two ways for localizing plates(yellow or other colors)
"""
def plate_detection(image):


	result = []

	rgbEdges = np.zeros(image.shape, dtype=np.uint8)

	hve = np.zeros(rgbEdges[:,:,0].shape, dtype=np.uint8)
	for plane in [0,2]:
		rgbEdges[:,:,plane] = cv2.Canny(image[:,:,plane], 50,200)
		hve = np.logical_or(rgbEdges[:,:,plane],hve)

	hve = np.uint8(hve)

	contours, hierarchy = cv2.findContours(hve, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	for i in range(len(contours)):

		if(len(contours[i]) > 3):

				rec = cv2.minAreaRect(contours[i])

				leg_ratio = rec[1][0] / rec[1][1]

				if(not (1/2 < leg_ratio < 2) and (1/5 < leg_ratio < 5)):
					recp = np.array(cv2.boxPoints(rec), dtype=np.int32)
					x_dis = np.max(recp[:, 0]) - np.min(recp[:, 0])
					y_dis = np.max(recp[:, 1]) - np.min(recp[:, 1])
					if(x_dis > 80 and y_dis > 10):
						if(all([0 < xCor < image[:,:,0].shape[1]  for xCor in recp[:,0]]) and  all([0 < yCor < image[:,:,0].shape[0]  for yCor in recp[:,1]])):

							if (rec[1][0] < rec[1][1]):
								rec = (rec[0], (rec[1][1], rec[1][0]), rec[2] + 90)

							if (-42 < rec[2] < 42):
								hull = cv2.convexHull(contours[i])
								rectness = cv2.contourArea(hull[:, 0, :]) / cv2.contourArea(recp)
								#print(rectness)
								if ( rectness > 0.80 ):
									#cv2.fillPoly(image, [recp], (255, 0, 0))
									rm = cv2.getRotationMatrix2D(tuple(rec[0]),rec[2],1)

									rotImg = cv2.warpAffine(src=image.copy(), M=rm ,dsize=(image[:,:,0].shape[1],image[:,:,0].shape[0]))

									xCor = int(rec[0][0] - rec[1][0] / 2) -1
									yCor = int(rec[0][1] - rec[1][1] / 2) -1

									cropped = rotImg[yCor: yCor + int(rec[1][1])+ 2,  xCor:xCor + int(rec[1][0]) + 2]

									result.append(cropped)

	#cv2.namedWindow("t", cv2.WINDOW_NORMAL)
	#cv2.imshow("t", image)
	#cv2.waitKey()

	return result
