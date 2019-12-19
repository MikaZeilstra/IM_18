import cv2
import numpy as np
import os
from heapq import nsmallest

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
	gray = cv2.cvtColor(plate_imgs, cv2.COLOR_BGR2GRAY)
	t = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 5)
	# print(np.average(gray))
	# gray = cv2.bilateralFilter(gray, 7, 21, 51)q

	#

	kernel = np.ones((2, 2), np.uint8)
	#t = cv2.morphologyEx(t, cv2.MORPH_OPEN, kernel,iterations=1)
	#t = cv2.morphologyEx(t, cv2.MORPH_CLOSE, kernel,iterations=1)
	edges = cv2.Canny(t, 50, 100)

	cont, hier = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	mask = np.zeros(plate_imgs.shape)
	c2 = []
	areadict = {}
	area = []
	for i in range(len(cont)):
		if( len(cont[i]) > 3 and hier[0][i][3] < 0):
			minRec = cv2.minAreaRect(cont[i])
			hull = cv2.convexHull(cont[i])
			brec = cv2.boundingRect(cont[i])
			if(brec[2] < brec[3] and  brec[2] > 5 and brec[3] > 5  and 1/2 < brec[2]/brec[3] < 1/1.1):
				areadict[brec[2] * brec[3]] = cont[i]
				area.append(brec[2] * brec[3])
				cv2.rectangle(edges,(brec[0],brec[1]),(brec[0] + brec[2],brec[1]+brec[3]),120,2)
				#c2.append(np.array(cv2.boxPoints(minRec),dtype=np.int32))

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


	cv2.drawContours(plate_imgs,c2, -1, 255, 1)
	cv2.imshow('frame', edges)
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
