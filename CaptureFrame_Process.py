import cv2
import numpy as np
import os
import pandas as pd
import Localization
import Recognize


THRESHOLD  = 0.725
"""
In this file, you will define your own CaptureFrame_Process funtion. In this function,
you need three arguments: file_path(str type, the video file), sample_frequency(second), save_path(final results saving path).
To do:
	1. Capture the frames for the whole video by your sample_frequency, record the frame number and timestamp(seconds).
	2. Localize and recognize the plates in the frame.(Hints: need to use 'Localization.plate_detection' and 'Recognize.segmetn_and_recognize' functions)
	3. If recognizing any plates, save them into a .csv file.(Hints: may need to use 'pandas' package)
Inputs:(three)
	1. file_path: video path
	2. sample_frequency: second
    3. save_path: final .csv file path
Output: None
"""

def incDiff (hl, hln, rcont):
    # avgColArr = []
    #
    # for x in range(hl.shape[0]) :
    #     for y in range(hl.shape[1]) :
    #         np.append(avgColArr, hln[x][y])
    #
    # avgCol = np.average(avgColArr);

    lbrt = 0.7
    print("LBRT: ", lbrt)

    for x in range(hl.shape[0]) :
        for y in range(hl.shape[1]) :
            hln[x][y] = hln[x][y] - lbrt


    for x in range(hl.shape[0]) :
        for y in range(hl.shape[1]) :
            hln[x][y] = hln[x][y] * rcont



def CaptureFrame_Process(file_path, sample_frequency, save_path):
    print("Now loading file " + file_path)
    cap = cv2.VideoCapture(file_path)
    ret, frame = cap.read()

    print(frame.shape)
    # Display the resulting frame
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #h = hsv[:,:,0]
    hl = cv2.Laplacian(hsv,cv2.CV_64F)
    hln = np.uint8(cv2.normalize(hl, None, 0, 255, cv2.NORM_MINMAX))
    binary = np.zeros(hln.shape[0] * hln.shape[1]).reshape(hln.shape[0:2])
    ge = cv2.cvtColor(hln, cv2.COLOR_BGR2GRAY)
    #ge = cv2.equalizeHist(g)

    t = cv2.adaptiveThreshold(hsv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 0)

    #contours, hierarchy = cv2.findContours(t, 1, 2)
    #t = cv2.drawContours(t ,contours, -1, (0,255,0),3)

    print(np.max(hln))

    # for x in range(hl.shape[0]) :
    #     for y in range(hl.shape[1]):
    #         if(hln[x][y][0] <=THRESHOLD and hln[x][y][1] <=THRESHOLD and hln[x][y][2] <=THRESHOLD):
    #             binary[x][y] = 0
    #         else:
    #             binary[x][y] = 1



    cv2.imshow('frame',t)

    cv2.waitKey()
    cap.release()