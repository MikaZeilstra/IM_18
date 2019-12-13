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
	# print(frame.shape)
	# Display the resulting frame
	hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	for x in range(len(hsv)):
		for y in range(len(hsv[x])):
			hsv[x][y][2] = 128;
			hsv[x][y][1] = 128;
	eimage = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
	#gray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
	#gray = np.uint8(hsv[:,:,0] * 0.5 + hsv[:,:,1]*0.5)
	gray = hsv[:,:,0]
	print(gray.shape)
	# h = hsv[:,:,0]
	# hl = cv2.Laplacian(hsv,cv2.CV_64F)
	# hln = np.uint8(cv2.normalize(hl, None, 0, 255, cv2.NORM_MINMAX))
	# binary = np.zeros(hln.shape[0] * hln.shape[1]).reshape(hln.shape[0:2])
	# ge = cv2.cvtColor(hln, cv2.COLOR_BGR2GRAY)
	# ge = cv2.equalizeHist(g)

	# t = cv2.adaptiveThreshold(hsv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 0)
	# tl = cv2.Laplacian(t,cv2.CV_8U)
	gray = cv2.GaussianBlur(gray,(3,3),10)
	gray = cv2.bilateralFilter(gray, 7, 21,51)
	edges = cv2.Canny(gray, 50 ,250)
	kernel = np.ones((100, 100), np.uint8)
	#edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
	edgesc = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)


	contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	c2 = []
	nfound = 0;

	mask2 = np.zeros(edges.shape, np.uint8)
	for i in range(len(contours)):
		#if (hierarchy[0][i][3] < 0 and hierarchy[0][i][2] > 0):
		if(hierarchy[0][i][3] < 0):
			nfound += 1;
			# peri = cv2.arcLength(contours[i], False)
			# aprox = cv2.approxPolyDP(contours[i],peri * 0.02,False)
			# if(len(aprox) == 4):


			rec = cv2.minAreaRect(contours[i])
			hull = cv2.convexHull(contours[i])
			recp = np.array(cv2.boxPoints(rec), dtype=np.int32)

			#cv2.fillPoly(edges, [hull[:, 0, :]], 255)


			area = cv2.contourArea(hull)
			circh = cv2.arcLength(hull, True)
			ratio = 0
			try:
				ratio = area / circh
			except:
				pass
			# if (250 <= len(contours[i]) <= 250):

			# ------------------------
			# Retrieve vaguely rectanglur stuff
			# -----------------------
			if (5 < ratio < 60 ):
				print(rec[2])
				mask = np.zeros(edges.shape, np.uint8)
				# print(hull)

				# cv2.drawContours()

				# print([recp])
				recp = np.array(cv2.boxPoints(rec), dtype=np.int32)
				print("found : " + str(i))
				leg_ratio = np.linalg.norm(recp[0] - recp[1]) / np.linalg.norm(recp[1] - recp[2])
				cv2.fillPoly(mask, [recp], 120)
				cv2.fillPoly(mask, [hull[:, 0, :]], 150)
				unique, counts = np.unique(mask, return_counts=True)

				#cv2.fillPoly(image, [recp], 120)
				#cv2.fillPoly(image, [hull[:, 0, :]], 255)

				#cv2.fillPoly(mask2, [recp], 120)
				#cv2.fillPoly(mask2, [hull[:, 0, :]], 255)

				x_dis = np.max(recp[:,0]) - np.min(recp[:,0])
				y_dis = np.max(recp[:,1]) - np.min(recp[:,1])

				if (counts[1] / counts[0] < 0.05 and not (1/2 < leg_ratio < 2) and (1/5 < leg_ratio < 5) and x_dis > y_dis and area > 100* 1/5 * 100):

					#cv2.circle(mask2, tuple(np.int32(rec[0])), 50, 255, 3)
					#print(recp)q
					# print(leg_ratio)
					# print(np.linalg.norm(recp[0] - recp[1]))
					# print(np.linalg.norm(recp[1] - recp[1]))
					# # print(cv2.contourArea([recp]) / cv2.contourArea(cv2.UMat(hull)))


					c2.append(contours[i])
					cv2.fillPoly(image, [recp], 120)
					cv2.fillPoly(image, [hull[:, 0, :]], 255)

					cv2.fillPoly(gray, [recp], 120)
					cv2.fillPoly(gray, [hull[:, 0, :]], 255)

			# defects = cv2.convexityDefects(contours[i], hulld)
			# defect_sum = np.sum(defects[:,0,3]/256)
			# print(defects)
			# print(defect_sum)
			# print(len(hull ))
			# cv2.rectangle(tc, (rec[0], rec[1]), (rec[0] + rec[2], rec[1] + rec[3]), 170, 2)
	# print(len(c2))

	# rec = cv2.boundingRect(c2[0])

	# print("Area : " + str(rec[2] * rec[3]))
	# print("circ : " + str(rec[2] * 2 + 2* rec[3]))
	# print((rec[2] * rec[3])/(rec[2] * 2 + 2 * rec[3]))



	#cv2.drawContours(mask ,c2, -1, 128,1)
	#plateTemplate = cv2.imread("TrainingSet/Templates/BinTemplate.jpg", cv2.IMREAD_GRAYSCALE)
	gray_edge = np.hstack((gray,edges))
	return image

# alg = cv2.createGeneralizedHoughBallard()
# alg.setTemplate(plateTemplate)
# pos, votes = alg.detect(t)
# print(pos[0][0][0:2])
# print(votes.shape)
#