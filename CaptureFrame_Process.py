import cv2
import numpy as np
from collections import Counter
import os
import pandas as pd
import Localization
import Recognize
import time
import DispatchQueue
import csv


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
    Symbols = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'B', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V','X', 'Z','-']

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
    #platesList = []
    #lTimes = []
    #rTimes = []
    #pFound = []
    start = time.time()
    q = DispatchQueue.disQueue(f_total)
    q.startWork()
    framen = 0
    while(True):
        framen += 1
        ret, frame = cap.read()
        #print(ret)
        if(not(ret)):
            #cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
            #continue
            break
        #if( 397< framen <504):
        if(False):
            q.addFrame([frame,framen])
        # Lstart = time.time();
        # plate = Localization.plate_detection(frame)
        # lTimes.append(time.time() - Lstart)
        # pFound.append(len(plate))
        # #print("time for loc : " + str(time.time() - Lstart))
        # #print(plate.shape)
        # Rstart = time.time();
        # #print(len(plate))
        # for im in plate:
        #
        #     rec = Recognize.segment_and_recognize(im, trIm)
        #
        #     if len(rec) != 0:
        #         platesList.append(rec)
        # rTimes.append(time.time() - Rstart)
            #pass
        #print(f_total)
        #cv2.imshow('frame', frame)
        #if cv2.waitKey(spf) & 0xFF == ord('q'):
        #      break


    #cv2.imwrite("BinTemplate.jpg", t)
    #print("average plate found : " + str(np.average(pFound)))
    #print("Time  avg for loc : " + str(np.average(lTimes)))
    #print("Time  avg for rec : " + str(np.average(rTimes)))

    cv2.destroyAllWindows()
    cap.release()

    platesList, stamps = q.getResult()

    #print(platesList)
    #print(stamps)
    #counter = Counter(platesList)
    #print(platesList)

    uniqe = list(platesList.keys())
    count = [len(platesList[x]) for x in uniqe]

    #Translate to symbols
    #print(uniqe)
    #uniqe = [[Symbols[s] for s in p] for p in uniqe]
    realPlatesIndexes = []

    plates = list(zip(uniqe,count))
    print("--- PRINTING PLATES ---")

    for i, j in enumerate(count):
         if count[i] >= 5:
             realPlatesIndexes.append(i)

    finalRealPlates = []
    #familyPlates = []
    foundFamilies = np.full(len(realPlatesIndexes), -1)
    #print(foundFamilies)
    #maxOccur = -1

    #maxPlate = []
    lastFam = 0
    #print(len(realPlatesIndexes))

    for i, j in enumerate(realPlatesIndexes):
        #finalRealPlates.append(uniqe[j])
        #familyPlates.append(j)
        #print("Adding " + str(plates[j]))
        if(foundFamilies[i] == -1):
            foundFamilies[i] = lastFam
            lastFam += 1
        for k in range(i+1,min(i+11, len(realPlatesIndexes))):
            if not (Diff(plates[j][0], plates[realPlatesIndexes[k]][0])):
                print("Adding " + str(plates[realPlatesIndexes[k]]))
                foundFamilies[k] = foundFamilies[i]

        # if len(familyPlates) > 1:
        #     print("WE HAVE NOISE")
        #     maxPlate = plates[familyPlates[0]]
        #     max = plates[familyPlates[0]][1]
        #     for k in range(len(familyPlates)):
        #         if k+1 < len(familyPlates):
        #             if plates[familyPlates[k]][1] < plates[familyPlates[k+1]][1] and plates[familyPlates[k+1]][1] > max :
        #                 print("INDEX: " + str(k+1))
        #                 max = plates[familyPlates[k+1]][1]
        #                 maxPlate = plates[familyPlates[k+1]]
        #                 print("MAX " + str(max))

        # else:
        #  maxPlate = plates[familyPlates[0]]

        #lastFam = lastFam + len(familyPlates)-2

        #familyPlates = []
        #print(maxPlate)
        #finalRealPlates.append(maxPlate)
        # print(maxOccur)
        # print(plates[realPlatesIndexes[maxOccur]])
        # finalRealPlates.append(plates[realPlatesIndexes[maxOccur]])


        #print(plates[realPlatesIndexes[maxOccur]])

        #maxOccur = 0
    print(list(zip(finalRealPlates,foundFamilies)))

    #finalRealPlates = unique_list(finalRealPlates)


    print("--------------FINAL RESULT------------")
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
                    # print(ffinalPlate)
                    # print(streak)
                    # print(finalPlate[i])

                if (streak >= 4):
                    ffinalPlate[-1] = 27
                    ffinalPlate.append(finalRealPlates[i][0][c - 1])
                    streak = 1
                ffinalPlate.append(finalRealPlates[i][0][c])
                prevchar = ffinalPlate[-1]
            finalRealPlates[i] = (ffinalPlate, finalRealPlates[i][1])
        ffinalPlate = [Symbols[s] for s in ffinalPlate]
        print()
        finalRealPlates[i] = ("".join(ffinalPlate),np.min(platesList[j[0]]),np.min(platesList[j[0]]) *spf /1000 /3)
        print("".join(ffinalPlate), finalRealPlates[i][1])

    data = pd.DataFrame.from_records(finalRealPlates,columns=["License plate", "Frame no.", "Timestamp(seconds)"])

    out = open(save_path,"w+",newline = '\n')
    data.to_csv(out, index=False)
    out.close()

    platesList = q.getResult()
    print("Total time taken : " + str(time.time() - start))

    # print(uniqe)
    # print("--------------REAL PLATES DETECTED------------")
    # for i, j in enumerate(realPlatesIndexes):
    #     print(plates[j])



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
    # intilize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
            # print list
    return unique_list
    # for x in unique_list:
    #     print
    #     x,