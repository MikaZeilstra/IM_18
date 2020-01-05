import cv2
import numpy as np
from collections import Counter
import os
import pandas as pd
import Localization
import Recognize
import time


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
    print("Now loading file " + file_path)
    Symbols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'B', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V','X', 'Z','-']

    trIm = []
    for y in range(0, 10):
        trainingImage = cv2.imread('SameSizeNumbers/' + str(y) + '.bmp', cv2.IMREAD_GRAYSCALE)  # trainImage
        trIm.append(trainingImage)
    # trIm.append(cv2.Canny(trainingImage, 50, 150))
    # kpt, dest = sift.detectAndCompute(trainingImage, None)
    # kp2.append(kpt)
    # des2.append(dest)

    for y in range(1, 18):
        trainingImage = cv2.imread('SameSizeLetters/' + str(y) + '.bmp', cv2.IMREAD_GRAYSCALE)  # trainImage
        trIm.append(trainingImage)

    cap = cv2.VideoCapture(file_path)
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO,1)
    t_total = cap.get(cv2.CAP_PROP_POS_MSEC)
    f_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
    spf = int(np.round(t_total/ f_total))
    print(spf)
    platesList = []
    lTimes = []
    rTimes = []
    pFound = []
    start = time.time()
    while(True):
        ret, frame = cap.read()
        #print(ret)
        if(not(ret)):
            #cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
            #continue
            break
        Lstart = time.time();
        plate = Localization.plate_detection(frame)
        lTimes.append(time.time() - Lstart)
        pFound.append(len(plate))
        #print("time for loc : " + str(time.time() - Lstart))
        #print(plate.shape)
        Rstart = time.time();
        #print(len(plate))
        for im in plate:

            rec = Recognize.segment_and_recognize(im, trIm)

            if len(rec) != 0:
                platesList.append(rec)
        rTimes.append(time.time() - Rstart)
            #pass
        #print(f_total)
        #cv2.imshow('frame', frame)
        #if cv2.waitKey(spf) & 0xFF == ord('q'):
        #      break

    #cv2.imwrite("BinTemplate.jpg", t)
    print("average plate found : " + str(np.average(pFound)))
    print("Time  avg for loc : " + str(np.average(lTimes)))
    print("Time  avg for rec : " + str(np.average(rTimes)))
    print("Total time taken : " + str(time.time() - start))
    cv2.destroyAllWindows()
    cap.release()


    counter = Counter(platesList)
    #print(platesList)

    uniqe = list(counter.keys())
    count = list(counter.values())

    #Translate to symbols
    #print(uniqe)
    uniqe = [[Symbols[s] for s in p] for p in uniqe]
    realPlatesIndexes = []

    plates = list(zip(uniqe,count))
    print("--- PRINTING PLATES ---")

    for i, j in enumerate(count):
         if count[i] >= 5:
             realPlatesIndexes.append(i)

    finalRealPlates = []
    familyPlates = []
    maxOccur = 0
    lastFam = 0
    print(len(realPlatesIndexes))

    for i, j in enumerate(realPlatesIndexes):
        if i < lastFam-1:
            continue
        familyPlates.append(j)
        for k, l in enumerate(realPlatesIndexes):
            if len(Diff(plates[j][0], plates[l][0])) <= 2 :
                print("Comparing " + str(plates[j][0]) + " AND " + str(plates[l][0]))
                familyPlates.append(l)


        if len(familyPlates) > 1:
            print("IM HERE")
            maxOccur = i
            for k in range(len(familyPlates)):
                #print(l)
                if k+1 != len(familyPlates):
                    if plates[familyPlates[k]][1] < plates[familyPlates[k+1]][1] :
                        maxOccur = k+1

                lastFam = lastFam + k
        else:
            maxOccur = maxOccur + 1

        print("-----------SEPARATOR-------------")
        print(maxOccur)
        print(plates[realPlatesIndexes[maxOccur]])
        finalRealPlates.append(plates[realPlatesIndexes[maxOccur]])
        #maxOccur = 0
        familyPlates = []

        #i = k

    #
    # for i in range(len(plates)):
    #     familyPlates.append(i)
    #     for j in range(len(plates)):
    #         if len(Diff(plates[i][0], plates[j][0])) <= 2 :
    #             familyPlates.append(j)
    #             #print(j)
    #
    #     for k, l in enumerate(familyPlates):
    #         print(l)
    #         # if plates[l][1] > plates[l][1] :
    #             #maxOccur = l
    #     print("-----------SEPARATOR-------------")
    #     finalRealPlates.append(plates[maxOccur])
    #     maxOccur = 0
    #     familyPlates = []
    #     i = j

    print("--------------FINAL RESULT------------")
    for i, j in enumerate(finalRealPlates) :
        #print()
        print(j)

    # print(uniqe)
    print("--------------REAL PLATES DETECTED------------")
    for i, j in enumerate(realPlatesIndexes):
        print(plates[j])



def Diff(li1, li2):
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
    return li_dif