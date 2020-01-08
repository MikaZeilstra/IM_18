import cv2
import numpy as np
from collections import Counter
import pandas as pd
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

    cap = cv2.VideoCapture(file_path)
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO,1)
    t_total = cap.get(cv2.CAP_PROP_POS_MSEC)
    f_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
    spf = int(np.round(t_total/ f_total))

    start = time.time()
    q = DispatchQueue.disQueue(f_total)
    q.startWork()
    framen = 0
    print("Now loading : " + file_path)
    while(True):
        framen += 1
        ret, frame = cap.read()
        if(not(ret)):
            break

        #if(540 < framen < 577):
        q.addFrame([frame,framen])

    cv2.destroyAllWindows()
    cap.release()

    platesList = q.getResult()
    lastFam = 0
    #print(platesList)
    platesList.sort(key=lambda x : x[1])

    foundFamilies = np.full(len(platesList), -1)
    for i, j in enumerate(platesList):
        if(foundFamilies[i] == -1):
            foundFamilies[i] = lastFam
            lastFam += 1
        for k in range(i+1, min(i+13, len(platesList))):
            #print(Diff(platesList[i][0], platesList[k][0]))
            if Diff(platesList[i][0], platesList[k][0]) <= 2:
                foundFamilies[k] = foundFamilies[i]

    families = [[]]

    for i, j in enumerate(foundFamilies):
        if(len(families) <= j):
            families.append([])
        families[j].append(platesList[i])

    for f in families[:]:
        if(len(f)) < 5 :
            families.remove(f)

    finalRealPlates = []
    for f in families[:]:
        #print(f)
        fFrame = min(f,key= lambda x: x[1])[1]
        count = Counter(list(map(lambda x : x[0], f)))
        plate = count.most_common(1)[0]
        finalRealPlates.append((plate[0],fFrame,plate[1]))

    for i, j in enumerate(finalRealPlates[:]) :
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
            finalRealPlates[i] = (ffinalPlate, finalRealPlates[i][1], finalRealPlates[i][2])
        ffinalPlate = [Symbols[s] for s in ffinalPlate]

        finalRealPlates[i] = ("".join(ffinalPlate), finalRealPlates[i][1], finalRealPlates[i][1] *spf /1000,finalRealPlates[i][2])

    data = pd.DataFrame.from_records(finalRealPlates,columns=["License plate", "Frame no.", "Timestamp(seconds)","Occurences"])

    out = open(save_path,"w+",newline = '\n')
    data.to_csv(out, index=False)
    out.close()


    print("")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(data)
    print("Total time taken : " + str(time.time() - start))


def Diff(li1, li2):
    n = len(li1)
    m = len(li2)

    me = np.zeros((n+1,m+1))

    for i in range(n+1):
        for j in range(m+1):
            if (i == 0):
                me[i][j] = j
            elif (j == 0):
                me[i][j] = i
            else:
                if (li1[i -1]!= li2[j-1]):
                    me[i][j] = min(min(1+me[i-1][j], 1+me[i-1][j-1]), 1+me[i][j -1] );
                else:
                    me[i][j] = me[i-1][j-1];
    #print(me)
    return me[n][m]
    # if len(li1) != len(li2) :
    #     return True;
    # else:
    #     diffCount = 0
    #     for i, j in enumerate(li1):
    #         if(j != li2[i]) :
    #             diffCount = diffCount + 1
    #
    #     if (diffCount > 2) :
    #         return True
    #     else:
    #         return False
