import cv2
import numpy as np
from skimage.io import imread, imshow

from find_maxima import *
from genHough.build_reference_table import *
from match_table import *

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
	sift = cv2.xfeatures2d.SIFT_create()
	kp2 = []
	des2 = []
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


	print(trIm)



	gray = cv2.cvtColor(plate_imgs, cv2.COLOR_BGR2GRAY)
	t = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 5)
	# print(np.average(gray))
	# gray = cv2.bilateralFilter(gray, 7, 21, 51)q

	#
	img3 = t

	kernel = np.ones((2, 2), np.uint8)
	#t = cv2.morphologyEx(t, cv2.MORPH_OPEN, kernel,iterations=1)
	#t = cv2.morphologyEx(t, cv2.MORPH_CLOSE, kernel,iterations=1)
	edges = cv2.Canny(t, 0, 100)



	newT = cv2.Laplacian(t, cv2.CV_8U)

	# cv2.imshow("T", newT)
	# cv2.waitKey()

	#edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones([2, 2]), iterations=2)

	cont, hier = cv2.findContours(newT, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	mask = np.zeros(plate_imgs.shape)
	c2 = []
	areadict = {}
	area = []
	for i in range(len(cont)):
		if( len(cont[i]) > 3 and hier[0][i][3] < 0):
			minRec = cv2.minAreaRect(cont[i])
			hull = cv2.convexHull(cont[i])
			brec = cv2.boundingRect(cont[i])
			if(brec[2] < brec[3] and  brec[2] > 10 and brec[3] > 10  and 1/2 < brec[2]/brec[3] < 1/1.1):
				cv2.rectangle(edges,(brec[0],brec[1]),(brec[0] + brec[2],brec[1]+brec[3]),120,2)

				allDiffs = []

				#for t in range(0, 10):

				for t in range(0, 27):
					#allDiffs.append([])

					areadict[brec[2] * brec[3]] = cont[i]
					area.append(brec[2] * brec[3])

					#croppedImage = edges[brec[1]:brec[1] + brec[3], brec[0]:brec[0] + brec[2]]

					croppedImage = edges[brec[1]:brec[1] + brec[3], brec[0]:brec[0] + brec[2]]

					#cv2.fillPoly(edges, pts=[cont[i]], color=(255,255,255))

					#cv2.fillConvexPoly(edges, cont[i], 255, lineType=True)

					#cv2.floodFill(edges, cont[i],  , 255)


					cv2.drawContours(edges, cont, i, 255,-1)


					#cv2.fillPoly(edges, cont[i], 255, lineType=True)





					image = trIm[t]

					croppedImage = cv2.morphologyEx(croppedImage, cv2.MORPH_OPEN, np.ones([3,3]), iterations=2)

					#print("SHAPE CROPPED: " + str(croppedImage.shape[0]))

					#print("SHAPE TRAIN: " + str(image.shape[0]))
					#
					# ratio1 = croppedImage.shape[0] / image.shape[0]
					# ratio2 = croppedImage.shape[1] / image.shape[1]

					width = int(croppedImage.shape[1])
					height = int(croppedImage.shape[0])
					dim = (width, height)
					# resize image
					resized = cv2.resize(image, dim, interpolation=cv2.INTER_NEAREST)





					#print("SHAPE RESIZED : " + str(resized.shape[0]))

					diff = np.logical_xor(resized, croppedImage, dtype=np.int16)

					diff = np.sum(diff) / (croppedImage.shape[0] * croppedImage.shape[1])

					allDiffs.append(diff)

					print("COMPARING WITH " + str(t) + " AND DIFF IS: " + str(diff))

					#print(diff.dtype)

					#print("DIFFERENCE : " + str(diff))



				allMins = []

				print(allDiffs)
				#for t in range(0, 10):

				minIndex = np.argmin(allDiffs)
				if (allDiffs[minIndex] < 0.3):
					print("WE FOUND A :" + str(minIndex))

				cv2.imshow("frame", edges)
				cv2.waitKey()


				# reference_image = trIm[5]
				# detect_s = GeneralisedHough.general_hough_closure(reference_image)
				#
				# im4 = edges[brec[1]:brec[1] + brec[3], brec[0]:brec[0] + brec[2]]
				# GeneralisedHough.test_general_hough(detect_s, reference_image, im4)

				# template = trIm[5]
				# im4 = edges[brec[1]:brec[1] + brec[3], brec[0]:brec[0] + brec[2]]
				# maskis, draw = GenHough.hough(im4, template)

				#im4 = edges[brec[1]:brec[1] + brec[3], brec[0]:brec[0] + brec[2]]


				# for y, g in enumerate(trIm):
				# 	refim = imread("genHough/Input1Ref.png")
				# 	im = edges #imread('Input1.png')
				#
				# 	table = buildRefTable(refim)
				# 	acc = matchTable(im, table)
				# 	val, ridx, cidx = findMaxima(acc)
				# 	# code for drawing bounding-box in accumulator array...
				#
				# 	acc[ridx - 5:ridx + 5, cidx - 5] = val
				# 	acc[ridx - 5:ridx + 5, cidx + 5] = val
				#
				# 	acc[ridx - 5, cidx - 5:cidx + 5] = val
				# 	acc[ridx + 5, cidx - 5:cidx + 5] = val
				#
				# 	plt.figure(1)
				# 	imshow(acc)
				# 	plt.show()
				#
				# 	# code for drawing bounding-box in original image at the found location...
				#
				# 	# find the half-width and height of template
				# 	hheight = np.floor(refim.shape[0] / 2) + 1
				# 	hwidth = np.floor(refim.shape[1] / 2) + 1
				#
				# 	# find coordinates of the box
				# 	rstart = int(max(ridx - hheight, 1))
				# 	rend = int(min(ridx + hheight, im.shape[0] - 1))
				# 	cstart = int(max(cidx - hwidth, 1))
				# 	cend = int(min(cidx + hwidth, im.shape[1] - 1))
				#
				# 	# draw the box
				# 	im[rstart:rend, cstart] = 255
				# 	im[rstart:rend, cend] = 255
				#
				# 	im[rstart, cstart:cend] = 255
				# 	im[rend, cstart:cend] = 255
				#
				# 	# show the image
				# 	plt.figure(2), imshow(refim)
				# 	plt.figure(3), imshow(im)
				# 	plt.show()


				# alg = cv2.createGeneralizedHoughBallard()
				#
				#
				#
				# alg.setTemplate(trIm[5])
				#
				# pos, votes = alg.detect(im4)
				#
				# for g, h in enumerate(pos):
				# 	if votes[g] > 5:
				# 		print("FOUND NUMBER " + str(5))




				#c2.append(np.array(cv2.boxPoints(minRec),dtype=np.int32))
				# for y in range(0, 10):
				# 	#trainingImage = cv2.imread('SameSizeNumbers/' + str(y) + '.bmp', cv2.IMREAD_GRAYSCALE)  # trainImage
				#
				#
				# 	# find the keypoints and descriptors with SIFT
				# 	im4 = edges[brec[1]:brec[1] + brec[3], brec[0]:brec[0] + brec[2]]
				# 	kp1, des1 = sift.detectAndCompute(im4, None)
				# 	#kp2, des2 = sift.detectAndCompute(trainingImage, None)
				#
				# 	# BFMatcher with default params
				# 	bf = cv2.BFMatcher()
				# 	matches = bf.knnMatch(des1, des2[y], k=2)
				#
				# 	# Apply ratio test
				# 	good = []
				# 	for m, n in matches:
				# 		if m.distance <  0.75 * n.distance:
				# 			good.append([m])
				#
				# 	if len(good) > 3:
				# 		print("FOUND ME A NUMBER YAY: " + str(y))
				# 		img3 = cv2.drawMatchesKnn(im4, kp1, trIm[y], kp2[y], good, None,
				# 								  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
				# 		plt.imshow(img3), plt.show()
				# 		for j in good:
				# 			print(j)
					# cv.drawMatchesKnn expects list of lists as matches.


	# print(area)
	# min_areas = []
	# min_var = 10**5
	# for ar in  area:
	# 	ns = nsmallest(4,area,key=lambda y : abs(ar-y))
	# 	if(np.var(ns) < min_var):
	# 		min_areas = ns
	# 		min_var = np.var(ns)
	# if(len(area) > 4):
	# 	for n in min_areas:
	# 		area.remove(n)
	# 	ns = add_smallest_var(ns,area)
	# 	while(np.var(ns) < np.var(min_areas) + np.mean(min_areas) and area):
	# 		print(min_areas)
	# 		min_areas = ns
	# 		ns = add_smallest_var(ns, area)
	# else:
	# 	return ""
	# #print(min_areas)
	# for ar in min_areas:
	# 	c2.append(areadict[ar])




	#cv2.drawContours(plate_imgs,c2, -1, 255, 1)
	#cv2.imshow('frame', edges)
	#cv2.imshow("frame", img3)
	cv2.waitKey()

	return ""

def add_smallest_var(to_add, options):
	min_var = 10**5
	res = []
	for o in options:
		if np.var(to_add + [o]) < min_var:
			res = to_add + [o]
			min_var = np.var(res)
	options.remove(res[-1])
	return res
