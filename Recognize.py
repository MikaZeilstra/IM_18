import cv2
import numpy as np
from collections import defaultdict
from scipy import stats as st

"""
In this file, you will define your own segment_and_recognize function.
To do:
	1. Segment the plates character by character
	2. Compute the distances between character images and reference character images(in the folder of 'SameSizeLetters' and 'SameSizeNumbers')
	3. Recognize the character by comparing the distances
Inputs:(One)
	1. plate_imgs: cropped plate images by Localization.plate_detection function
	type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
Outputs:(One)
	1. recognized_plates: recognized plate characters
	type: list, each element in recognized_plates is a list of string(Hints: the element may be None type)
Hints:
	You may need to define other functions.
"""
def segment_and_recognize(plate_imgs,trIm):
	#print(plate_imgs)
	#cv2.namedWindow("pl" , cv2.WINDOW_NORMAL)
	#cv2.imshow("pl", plate_imgs)
	#cv2.waitKey()


	# Initiate SIFT detector
	# kp2 = []
	# des2 = []



	#print(trIm)
	dashes = False


	gray = cv2.cvtColor(plate_imgs, cv2.COLOR_BGR2GRAY)
	for it in range(3):
		gray = cv2.bilateralFilter(gray, -1, 10, 11)

	t = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 5)
	# print(np.average(gray))
	# gray = cv2.bilateralFilter(gray, 7, 21, 51)q

	#
	#img3 = t

	# kernel = np.ones((2, 2), np.uint8)
	#t = cv2.morphologyEx(t, cv2.MORPH_OPEN, kernel,iterations=1)
	#t = cv2.morphologyEx(t, cv2.MORPH_CLOSE, kernel,iterations=1)
	#newT = cv2.Canny(gray, 100, 200)



	newT = cv2.Laplacian(t, cv2.CV_8U)

	#cv2.imshow("T", newT)
	#cv2.waitKey()

	#edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones([2, 2]), iterations=2)

	cont, hier = cv2.findContours(newT, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	# mask = np.zeros(plate_imgs.shape)
	#c2 = []
	checked = []
	brecs = {}
	contIds = []
	# areadict = {}
	# area = []

	brecVals = []
	plate = []
	dashid = []
	for i in range(len(cont)):
		if( len(cont[i]) > 3 ):
			#minRec = cv2.minAreaRect(cont[i])
			#hull = cv2.convexHull(cont[i])
			brec = cv2.boundingRect(cont[i])
			#cv2.rectangle(newT, (brec[0], brec[1]), (brec[0] + brec[2], brec[1] + brec[3]), 120, 1)
			if(brec[2] < brec[3] and  brec[2] > 5 and brec[3] > 10 and [brec[0],brec[1]] not in checked ):
				checked.append([brec[0],brec[1]])
				brecs[i] = brec
				contIds.append(i)

				#cv2.circle(edges, (brec[0],brec[1]), 10, 170, 2)
				#mask = np.zeros(edges.shape)
				#cv2.drawContours(mask,cont,i,255)
				#print()
				# cv2.imshow("", mask)
				# cv2.waitKey()
			elif ( brec[2] < plate_imgs.shape[1]/3 and brec[2] < plate_imgs.shape[0]/3 and 1.1 >  brec[3] / brec[2] > 1/3.5 and [brec[0],brec[1]] not in checked ):
				checked.append([brec[0], brec[1]])
				dashid.append(i)
				brecs[i] = brec
	#children = []
	#childMap = defaultdict(list)
	for id in contIds[:]:
	 	#print(hier[0][id][2])
	 	#print(id)
	 	isInside = check_inside(brecs,brecs[id] , id)
	 	if isInside:
	 		#children.append(cont[id])
	 		#childMap[isInside - 1].append(cont[id])
	 		contIds.remove(id)
	 		del brecs[id]
	# 		#print("t")
	# 		#cv2.drawContours(newT, cont, id, 90, -1)


	if(len(brecs) > 0):
		mode_Y = st.mode(np.array(list(brecs.values()))[:,1])[0]
		mode_High = st.mode(np.array(list(brecs.values()))[:, 3])[0]
		eps = np.ceil(plate_imgs.shape[0] /10)

		#print(mode_Y)
		#print(eps)

		for id in contIds[:]:
			if not(mode_Y - eps < brecs[id][1] < mode_Y + eps and mode_High - eps < brecs[id][3] < mode_High + eps):
				contIds.remove(id)
				del brecs[id]
		for id in dashid[:]:
			if not(mode_Y - eps < brecs[id][1] and mode_High + eps > brecs[id][3]):
				dashid.remove(id)
				del brecs[id]
	else:
		return []

	for id in dashid:
		croppedImage = newT[brecs[id][1]:brecs[id][1] + brecs[id][3], brecs[id][0]:brecs[id][0] + brecs[id][2]].copy()
		cv2.floodFill(croppedImage, None, (int(croppedImage.shape[1] / 2) - 1, int(croppedImage.shape[0] / 2) - 1), 255)
		# print((int(croppedImage.shape[1]/2),int(croppedImage.shape[0]/2)))
		# cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
		# cv2.imshow("frame", croppedImage)
		# cv2.waitKey()
		if (np.average(croppedImage) > 0.6 * 255):
			dashes = True
			#print(dashes)



	for id in contIds:
		allDiffs = []

		#cv2.rectangle(newT, (brecs[id][0], brecs[id][1]),(brecs[id][0] + brecs[id][2], brecs[id][1] + brecs[id][3]), 120, 1)
		#print(brecs[id][1])
		# for t in range(0, 10):
		croppedImage = newT[brecs[id][1]:brecs[id][1] + brecs[id][3], brecs[id][0]:brecs[id][0] + brecs[id][2]]
		#croppedImage = cv2.copyMakeBorder(croppedImage, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

		ccontours, chier = cv2.findContours(croppedImage,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

		cropchecked = []
		cbrecs = {}
		ccontIds = []

		for ci in range(len(ccontours)):
			cbrec = cv2.boundingRect(ccontours[ci])
			if([cbrec[0],cbrec[1]] not in cropchecked):
				cropchecked.append([cbrec[0],cbrec[1]])
				cbrecs[ci] = cbrec
				ccontIds.append(ci)
				#cv2.rectangle(croppedImage, (cbrecs[ci][0], cbrecs[ci][1]), (cbrecs[ci][0] + cbrecs[ci][2], cbrecs[ci][1] + cbrecs[ci][3]), 120, 1)

		cropchildMap = defaultdict(list)





		for cid in ccontIds[:]:
			isInside = check_inside(cbrecs, cbrecs[cid], cid)
			if isInside:
				#print("c")
				cropchildMap[isInside - 1].append(ccontours[cid])
				ccontIds.remove(cid)
				del cbrecs[cid]





		#symbol = np.zeros(croppedImage.shape)
		for cid in ccontIds :
			#pass

			cv2.drawContours(croppedImage,ccontours,cid,255,-1)
			cv2.drawContours(croppedImage, cropchildMap[cid], -1, 0, -1)




		#print(len(cropchildMap[ccontIds[0]]))


		#mask = np.zeros(croppedImage.shape)


		#cv2.drawContours(newT, cont, id, 255, -1)
		#cv2.drawContours(newT, childMap[id], -1,0,-1)
		croppedImage = cv2.morphologyEx(croppedImage, cv2.MORPH_OPEN, np.ones([2, 2]), iterations=1)





		width = int(croppedImage.shape[1])
		height = int(croppedImage.shape[0])
		dim = (width, height)



		for t in range(0, 27):
			image = trIm[t]

			# resize image
			resized = cv2.resize(image, dim, interpolation=cv2.INTER_NEAREST)

			# print("SHAPE RESIZED : " + str(resized.shape[0]))

			diff = np.logical_xor(resized, croppedImage, dtype=np.int16)

			diff = np.sum(diff) / (croppedImage.shape[0] * croppedImage.shape[1])

			allDiffs.append(diff)

			#print("COMPARING WITH " + str(t) + " AND DIFF IS: " + str(diff))
			#print("DIFFERENCE : " + str(diff))
			#cv2.namedWindow("L", cv2.WINDOW_NORMAL)
			#cv2.imshow("L", np.hstack([croppedImage, resized]))
			#cv2.waitKey()

		#cv2.namedWindow("f",cv2.WINDOW_NORMAL)
		#cv2.imshow("f", croppedImage)
		#cv2.waitKey()
#
		minIndex = np.argmin(allDiffs)

		#print("found possible " + str(minIndex) + " with diff " + str(allDiffs[minIndex]))
		if (allDiffs[minIndex] < 0.35):
			#pass
			#print("WE FOUND A :" + str(minIndex))
			brecVals.append(brecs[id][0])
			# print("APPENDING: " + str(brec[0]))
			plate.append(minIndex)


	finalPlate = []

	#cv2.namedWindow("newT", cv2.WINDOW_NORMAL)
	#cv2.imshow("newT", newT)
	#cv2.waitKey()

	#print(brecVals)




	valsLength = len(brecVals)

	sortedMin = np.argsort(brecVals)

	#print(sortedMin)

	for i in range(0, valsLength) :
		#minX = np.argmin(brecVals)

		finalPlate.append(plate[sortedMin[i]])
		#brecVals.remove(brecVals[minX])
		#plate.remove(plate[minX])


	#print("LENGTH OF THE PLATE IS: " +str(len(finalPlate)))
	if len(finalPlate) >= 6:
		#print("PRINTINT LICENTE PLATE : ----------------------------")

		#print(finalPlate)
		#print(dashes)
		ffinalPlate = []
		prevchar = -1
		streak = 0
		for char in finalPlate[:] :
			if((0 <= prevchar <= 9 and char >= 10) or (0 <= char <= 9 and prevchar >= 10) and dashes):
				ffinalPlate.append(27)
				streak = 0
			else:
				streak += 1
			if(streak >= 4):
				ffinalPlate[-1] = 27
				ffinalPlate.append(prevchar)
			ffinalPlate.append(char)
			prevchar = char
			#print(ffinalPlate)
		return tuple(ffinalPlate)
	else:
		return []
	#cv2.drawContours(plate_imgs,c2, -1, 255, 1)
	#cv2.imshow('frame', edges)
	#cv2.imshow("frame", img3)
	#cv2.waitKey()

	#return finalPlate

def check_inside(br_list,check, br_id):
	#print(br_list)
	for id , br in br_list.items():
		if check[0] >= br[0] and check[1] >= br[1] and check[0] + check[2] <= br[0] + br[2] and check[1] + check[3] <= br[3] + br[3] and br_id != id:
			return id + 1
	return False
