import cv2
import numpy as np
import os

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
	# gray = cv2.bilateralFilter(gray, 7, 21, 51)

	#

	kernel = np.ones((2, 2), np.uint8)
	t = cv2.morphologyEx(t, cv2.MORPH_OPEN, kernel)
	t = cv2.morphologyEx(t, cv2.MORPH_CLOSE, kernel)
	edges = cv2.Canny(t, 50, 100)

	cont, hier = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	mask = np.zeros(plate_imgs.shape)
	c2 = []
	for i in range(len(cont)):
		minRec = cv2.minAreaRect(cont[i])
		brec = cv2.boundingRect(cont[i])
		if(brec[2] < brec[3]):
			c2.append(cont[i])

	cv2.drawContours(mask, c2, -1, 255, 1)
	cv2.imshow('frame', mask)
	cv2.waitKey()

	return ""