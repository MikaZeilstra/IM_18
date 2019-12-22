import cv2
import numpy as np

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
def segment_and_recognize(plate_imgs):
	# Initiate SIFT detector
	# kp2 = []
	# des2 = []
	trIm = []
	for y in range(0, 10):
		trainingImage = cv2.imread('SameSizeNumbers/' + str(y) + '.bmp', cv2.IMREAD_GRAYSCALE)  # trainImage
		trIm.append(trainingImage)
		#trIm.append(cv2.Canny(trainingImage, 50, 150))
		#kpt, dest = sift.detectAndCompute(trainingImage, None)
		#kp2.append(kpt)
		#des2.append(dest)

	for y in range(1, 18):
		trainingImage = cv2.imread('SameSizeLetters/' + str(y) + '.bmp', cv2.IMREAD_GRAYSCALE)  # trainImage
		trIm.append(trainingImage)


	#print(trIm)



	gray = cv2.cvtColor(plate_imgs, cv2.COLOR_BGR2GRAY)
	t = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 5)
	# print(np.average(gray))
	# gray = cv2.bilateralFilter(gray, 7, 21, 51)q

	#
	img3 = t

	# kernel = np.ones((2, 2), np.uint8)
	#t = cv2.morphologyEx(t, cv2.MORPH_OPEN, kernel,iterations=1)
	#t = cv2.morphologyEx(t, cv2.MORPH_CLOSE, kernel,iterations=1)
	edges = cv2.Canny(t, 0, 100)



	newT = cv2.Laplacian(t, cv2.CV_8U)

	# cv2.imshow("T", newT)
	# cv2.waitKey()

	#edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones([2, 2]), iterations=2)

	cont, hier = cv2.findContours(newT, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	# mask = np.zeros(plate_imgs.shape)
	c2 = []
	checked = []
	# areadict = {}
	# area = []

	brecVals = []
	plate = []
	for i in range(len(cont)):
		if( len(cont[i]) > 3 ):
			minRec = cv2.minAreaRect(cont[i])
			hull = cv2.convexHull(cont[i])
			brec = cv2.boundingRect(cont[i])

			if(brec[2] < brec[3] and  brec[2] > 10 and brec[3] > 10  and 1/2 < brec[2]/brec[3] < 1/1.1 and [brec[0], brec[1]] not in checked ):

				checked.append([brec[0],brec[1]])
				#cv2.rectangle(edges, (brec[0], brec[1]), (brec[0] + brec[2], brec[1] + brec[3]), 120, 2)
				#cv2.circle(edges, (brec[0],brec[1]), 10, 170, 2)
				mask = np.zeros(edges.shape)
				cv2.drawContours(mask,cont,i,255)
				#print()
				# cv2.imshow("", mask)
				# cv2.waitKey()
				allDiffs = []


				#for t in range(0, 10):
				croppedImage = edges[brec[1]:brec[1] + brec[3], brec[0]:brec[0] + brec[2]]

				cv2.drawContours(edges, cont, i, 255, -1)

				croppedImage = cv2.morphologyEx(croppedImage, cv2.MORPH_OPEN, np.ones([3, 3]), iterations=2)

				# cv2.imshow("f", croppedImage)
				# cv2.waitKey()

				width = int(croppedImage.shape[1])
				height = int(croppedImage.shape[0])
				dim = (width, height)




				for t in range(0, 27):
					image = trIm[t]
					#allDiffs.append([])

					# areadict[brec[2] * brec[3]] = cont[i]
					# area.append(brec[2] * brec[3])

					#croppedImage = edges[brec[1]:brec[1] + brec[3], brec[0]:brec[0] + brec[2]]


					#cv2.fillPoly(edges, pts=[cont[i]], color=(255,255,255))

					#cv2.fillConvexPoly(edges, cont[i], 255, lineType=True)

					#cv2.floodFill(edges, cont[i],  , 255)





					#cv2.fillPoly(edges, cont[i], 255, lineType=True)







					#print("SHAPE CROPPED: " + str(croppedImage.shape[0]))

					#print("SHAPE TRAIN: " + str(image.shape[0]))
					#
					# ratio1 = croppedImage.shape[0] / image.shape[0]
					# ratio2 = croppedImage.shape[1] / image.shape[1]


					# resize image
					resized = cv2.resize(image, dim, interpolation=cv2.INTER_NEAREST)





					#print("SHAPE RESIZED : " + str(resized.shape[0]))

					diff = np.logical_xor(resized, croppedImage, dtype=np.int16)

					diff = np.sum(diff) / (croppedImage.shape[0] * croppedImage.shape[1])

					allDiffs.append(diff)


					#print("COMPARING WITH " + str(t) + " AND DIFF IS: " + str(diff))

					#print(diff.dtype)

					#print("DIFFERENCE : " + str(diff))



				#allMins = []

				#print(allDiffs)
				#for t in range(0, 10):

				minIndex = np.argmin(allDiffs)
				if (allDiffs[minIndex] < 0.3):
					pass
					print("WE FOUND A :" + str(minIndex))
					brecVals.append(brec[0])
					print("APPENDING: " + str(brec[0]))
					plate.append(minIndex)


				# cv2.imshow("frame", edges)
				# cv2.waitKey()


	finalPlate = []


	valsLength = len(brecVals)

	sortedMin = np.argsort(brecVals)

	for i in range(0, valsLength) :
		#minX = np.argmin(brecVals)

		finalPlate.append(plate[sortedMin[i]])
		#brecVals.remove(brecVals[minX])
		#plate.remove(plate[minX])


	print("LENGTH OF THE PLATE IS: " +str(len(finalPlate)))
	if len(finalPlate) >= 6:

		print("PRINTINT LICENTE PLATE : ----------------------------")
		print(finalPlate)
		return finalPlate
	else:
		return []

	#cv2.drawContours(plate_imgs,c2, -1, 255, 1)
	#cv2.imshow('frame', edges)
	#cv2.imshow("frame", img3)
	#cv2.waitKey()

	#return finalPlate

