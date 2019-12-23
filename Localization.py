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
	result = []
	# hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	# for x in range(len(hsv)):
	# 	for y in range(len(hsv[x])):
	# 		pass
	# 		#hsv[x][y][2] = 128;
	# 		#hsv[x][y][1] = 128;
	# lGray = cv2.Laplacian(hsv[:,:,2], cv2.CV_8U)
	# eimage = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
	# eGray = cv2.cvtColor(eimage, cv2.COLOR_BGR2GRAY)
	# graySc = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# gray = np.uint8(hsv[:,:,0] * 0.5 + hsv[:,:,1]*0.5)
	# gray = hsv[:,:,1]
	# hs = np.hstack([hsv[:,:,0] , hsv[:,:,1],hsv[:,:,2]])
	rgbEdges = np.zeros(image.shape, dtype=np.uint8)
	#edgeFuse = np.zeros(image.shape[0:2],dtype=np.uint8)
	#print(edgeFuse.shape)
	hve = np.zeros(rgbEdges[:,:,0].shape, dtype=np.uint8)
	for plane in [0,2]:
		rgbEdges[:,:,plane] = cv2.Canny(image[:,:,plane], 50,200)
		hve = np.logical_or(rgbEdges[:,:,plane],hve)
		#np.sum([rgbEdges[:,:,plane],edgeFuse], where=[True, False])
		#np.add(edgeFuse,rgbEdges[:,:,plane],where=edgeFuse[edgeFuse==0])
	#hse = np.hstack([rgbEdges[:, :, 0], rgbEdges[:, :, 1], rgbEdges[:, :, 2]])
	#print(gray.shape)
	# h = hsv[:,:,0]
	hve = np.uint8(hve)
	#hl = cv2.Laplacian(image,cv2.CV_64F)
	# hln = np.uint8(cv2.normalize(hl, None, 0, 255, cv2.NORM_MINMAX))
	# binary = np.zeros(hln.shape[0] * hln.shape[1]).reshape(hln.shape[0:2])
	# ge = cv2.cvtColor(hln, cv2.COLOR_BGR2GRAY)
	#ge = cv2.equalizeHist(graySc)


	#t = cv2.adaptiveThreshold(graySc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 0)
	#tl = cv2.Laplacian(t,cv2.CV_8U)
	# gray = cv2.GaussianBlur(gray,(3,3),10)
	#gray = cv2.bilateralFilter(gray, 7, 21,51)
	#edges = cv2.Canny(graySc, 0,200)
	#kernel = np.ones((100, 100), np.uint8)
	#edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
	#edgesc = cv2.cvtColor(graySc,cv2.COLOR_GRAY2BGR)

	#hve = np.uint32(np.add(rgbEdges[:, :, 0],rgbEdges[:, :, 2]))

	contours, hierarchy = cv2.findContours(hve, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	c2 = []
	areas = []

	mask2 = np.zeros(image[:,:,0].shape, np.uint8)
	for i in range(len(contours)):
		#if (hierarchy[0][i][3] < 0 and hierarchy[0][i][2] > 0):
		if(hierarchy[0][i][3] < 0):
			# peri = cv2.arcLength(contours[i], False)
			# aprox = cv2.approxPolyDP(contours[i],peri * 0.02,False)
			# if(len(aprox) == 4):


			rec = cv2.minAreaRect(contours[i])
			hull = cv2.convexHull(contours[i])
			#recp = np.array(cv2.boxPoints(rec), dtype=np.int32)

			#cv2.fillPoly(edges, [hull[:, 0, :]], 255)


			#area = cv2.contourArea(hull)
			# circh = cv2.arcLength(hull, True)
			# ratio = 0
			# try:
			# 	ratio = area / circh
			# except:
			# 	pass
			# if (250 <= len(contours[i]) <= 250):

			# ------------------------
			# Retrieve vaguely rectanglur stuff
			# -----------------------
			if (len(hull) > 3):
				#print(rec[2])
				leg_ratio = rec[1][0] / rec[1][1]

				if(not (1/3.5 < leg_ratio < 3.5) and (1/5 < leg_ratio < 5)):
					recp = np.array(cv2.boxPoints(rec), dtype=np.int32)
					x_dis = np.max(recp[:, 0]) - np.min(recp[:, 0])
					y_dis = np.max(recp[:, 1]) - np.min(recp[:, 1])
					if(x_dis > 100 and y_dis > 20):

						mask = np.zeros(image[:,:,0].shape, np.uint8)
				# print(hull)

				# cv2.drawContours()

				# print([recp])

				#print("found : " + str(i))
						cv2.fillPoly(mask, [recp], 120)
						cv2.fillPoly(mask, [hull[:, 0, :]], 150)
						unique, counts = np.unique(mask, return_counts=True)


				#cv2.fillPoly(image, [recp], 120)
				#cv2.fillPoly(image, [hull[:, 0, :]], 255)

				# cv2.fillPoly(mask2, [recp], 120)
				# cv2.fillPoly(mask2, [hull[:, 0, :]], 255)

				#c2.append(contours[i])


						if (len(counts) > 2 and counts[1]/counts[2] < 0.25):
					#print(rec)
					#cv2.circle(mask2, tuple(np.int32(rec[0])), 50, 255, 3)
					#print(recp)q
					# print(leg_ratio)
					# print(np.linalg.norm(recp[0] - recp[1]))
					# print(np.linalg.norm(recp[1] - recp[1]))
					# # print(cv2.contourArea([recp]) / cv2.contourArea(cv2.UMat(hull)))
					#print(counts)
							if(rec[1][0] < rec[1][1]):
								rec = (rec[0],(rec[1][1],rec[1][0]),rec[2] + 90)

							if(-42 < rec[2] < 42):
								c2.append(contours[i])

								#cv2.fillPoly(image, [recp], 120)
								#cv2.fillPoly(image, [hull[:, 0, :]], 255)

					# cv2.fillPoly(gray, [recp], 120)
					# cv2.fillPoly(gray, [hull[:, 0, :]], 255)

								br = cv2.boundingRect(hull)

								areas.append(br[2] * br[3])

								rm = cv2.getRotationMatrix2D(tuple(rec[0]),rec[2],1)
							#(int(np.ceil(rec[1][0])),int(np.ceil(rec[1][1])))
								image = cv2.warpAffine(image, rm , image[:,:,0].shape,cv2.INTER_CUBIC)
								cropped = image[br[1]:br[1] + br[3], br[0]:br[0] + br[2]]
								result.append(cropped)
								#return cropped

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
	#areaS = np.argsort(areas)

	#c3 = []

	#for i in range(0,4) :
	#	c3.append(c2[areaS[i]])

	#cv2.drawContours(mask2 ,c2, -1, 128,1)
	#plateTemplate = cv2.imread("TrainingSet/Templates/BinTemplate.jpg", cv2.IMREAD_GRAYSCALE)
	# gray_edge = np.hstack((gray,edges))
	return result

# alg = cv2.createGeneralizedHoughBallard()
# alg.setTemplate(plateTemplate)
# pos, votes = alg.detect(tq)
# print(pos[0][0][0:2])
# print(votes.shape)
#