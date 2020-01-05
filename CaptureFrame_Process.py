import cv2
import numpy as np
from collections import Counter
import os
import pandas as pd
import Localization
import Recognize
import time
import DispatchQueue


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


def CaptureFrame_Process(file_path, sample_frequency, save_path):
    Symbols = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'B', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V','X', 'Z','-']

    trIm = []
    for y in range(0, 10):
        trainingImage = cv2.imread('SameSizeNumbers/' + str(y) + '.bmp', cv2.IMREAD_GRAYSCALE)  # trainImage
        trIm.append(trainingImage)

    for y in range(1, 18):
        trainingImage = cv2.imread('SameSizeLetters/' + str(y) + '.bmp', cv2.IMREAD_GRAYSCALE)  # trainImage
        trIm.append(trainingImage)

    cap = cv2.VideoCapture(file_path)
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO,1)
    t_total = cap.get(cv2.CAP_PROP_POS_MSEC)
    f_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
    spf = int(np.round(t_total/ f_total))
    platesList = []
    lTimes = []
    rTimes = []
    pFound = []
    start = time.time()
    q = DispatchQueue.disQueue(f_total)
    q.startWork()
    framen = 0
    while(True):
        framen += 1
        ret, frame = cap.read()
        if(not(ret)):
            break
        q.addFrame([frame,framen])

    cv2.destroyAllWindows()
    cap.release()

    platesList = q.getResult()


    uniqe = platesList.keys()
    count = [len(platesList[x]) for x in uniqe]

    realPlatesIndexes = []

    plates = list(zip(uniqe,count))

    for i, j in enumerate(count):
         if count[i] >= 5:
             realPlatesIndexes.append(i)

    finalRealPlates = []
    familyPlates = []

    maxPlate = []
    lastFam = 0

    for i, j in enumerate(realPlatesIndexes):
        if i < lastFam:
            continue
        familyPlates.append(j)
        for k, l in enumerate(realPlatesIndexes):
            if l == j :
                continue
            if not (Diff(plates[j][0], plates[l][0])):
                familyPlates.append(l)

        if len(familyPlates) > 1:
            maxPlate = plates[familyPlates[0]]
            max = plates[familyPlates[0]][1]
            for k in range(len(familyPlates)):
                if k+1 < len(familyPlates):
                    if plates[familyPlates[k]][1] < plates[familyPlates[k+1]][1] and plates[familyPlates[k+1]][1] > max :
                        max = plates[familyPlates[k+1]][1]
                        maxPlate = plates[familyPlates[k+1]]
        else:
            maxPlate = plates[familyPlates[0]]

        lastFam = lastFam + len(familyPlates)-2
        familyPlates = []
        finalRealPlates.append(maxPlate)


    finalRealPlates = unique_list(finalRealPlates)


    for i, j in enumerate(finalRealPlates[:]) :
        #print()
        ffinalPlate = finalRealPlates[i][0]
        if(finalRealPlates[i][0][-1] == 27):
            ffinalPlate = []
            prevchar = -1
            streak = 0
            for c in range(len(finalRealPlates[i][0][:])):
                if(finalRealPlates[i][0][c] == 27):
                    continue
                if ((0 <= prevchar <= 9 and 26 >= finalRealPlates[i][0][c] >= 10) or (
                        0 <= finalRealPlates[i][0][c] <= 9 and prevchar >= 10)):
                    ffinalPlate.append(27)
                    streak = 1
                else:
                    streak += 1

                if (streak >= 4):
                    ffinalPlate[-1] = 27
                    ffinalPlate.append(finalRealPlates[i][0][c - 1])
                    streak = 1
                ffinalPlate.append(finalRealPlates[i][0][c])
                prevchar = ffinalPlate[-1]
            finalRealPlates[i] = (ffinalPlate, finalRealPlates[i][1])
        ffinalPlate = [Symbols[s] for s in ffinalPlate]

        finalRealPlates[i] = ("".join(ffinalPlate),np.min(platesList[j[0]]),np.min(platesList[j[0]]) *spf /1000)

    data = pd.DataFrame.from_records(finalRealPlates,columns=["License plate", "Frame no.", "Timestamp(seconds)"])

    out = open(save_path,"w+",newline = '\n')
    data.to_csv(out, index=False)


def Diff(li1, li2):
    if len(li1) != len(li2) :
        return True;
    else:
        diffCount = 0
        for i, j in enumerate(li1):
            if(j != li2[i]) :
                diffCount = diffCount + 1

        if (diffCount > 2) :
            return True
        else:
            return False


def unique_list(list1):
    unique_list = []

    for x in list1:
        if x not in unique_list:
            unique_list.append(x)

    return unique_list
