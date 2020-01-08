import cv2
import numpy as np
from collections import defaultdict
from scipy import stats as st
from itertools import repeat

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


def segment_and_recognize(plate_imgs, trIm):
    dashes = False

    if plate_imgs.shape[0]*3 * plate_imgs.shape[1]*3 == 0 :
        return []

    plate_imgs = cv2.resize(plate_imgs,(plate_imgs.shape[1] * 3, plate_imgs.shape[0]*3),interpolation=cv2.INTER_NEAREST)

    gray = cv2.cvtColor(plate_imgs, cv2.COLOR_BGR2GRAY)

    #repeat(cv2.bilateralFilter(gray, -1, 5, 11),3)

    Tarea = int(plate_imgs.shape[1]*plate_imgs.shape[0]/400)

    #print(Tarea)

    t = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, Tarea if Tarea % 2 == 1 else Tarea +1, 5)
    newT = cv2.Laplacian(t, cv2.CV_8U)
    #newT = cv2.Canny(gray,110, 125)

    # cv2.namedWindow("t", cv2.WINDOW_NORMAL)
    # cv2.imshow("t",newT)
    # cv2.waitKey()


    cont, hier = cv2.findContours(newT, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    checked = []
    brecs = {}
    dbrecs = {}
    contIds = []

    brecVals = []
    plate = []
    dashid = []
    for i in range(len(cont)):
        if (len(cont[i]) > 3):
            brec = cv2.boundingRect(cont[i])

            if (brec[2] < brec[3] and brec[2] > 5 and brec[3] > 10 and [brec[0], brec[1]] not in checked):

                checked.append([brec[0], brec[1]])
                brecs[i] = brec
                contIds.append(i)

            elif (brec[2] < plate_imgs.shape[1] / 3 and brec[2] < plate_imgs.shape[0] / 3 and 1.1 > brec[3] / brec[2] > 1 / 3.5 and [brec[0], brec[1]] not in checked):
                checked.append([brec[0], brec[1]])
                dashid.append(i)
                dbrecs[i] = brec

    for id in contIds[:]:

        isInside = check_inside(brecs, brecs[id], id)
        if isInside:
            contIds.remove(id)
            del brecs[id]


    #print(eps)
    if (len(brecs) > 0):

        mode_Y = st.mode(np.array(list(brecs.values()))[:, 1])[0]
        mode_High = st.mode(np.array(list(brecs.values()))[:, 3])[0]
        eps = int(np.ceil(plate_imgs.shape[0] / 10))

        for id in contIds[:]:

            if not (mode_Y - eps <  brecs[id][1] < mode_Y + eps and mode_High - eps < brecs[id][3] < mode_High + eps):
                contIds.remove(id)
                del brecs[id]
        for id in dashid[:]:
            if not (mode_Y - eps < dbrecs[id][1] and mode_High + eps > dbrecs[id][3]):
                dashid.remove(id)
                del dbrecs[id]
    else:
        return []

    for id in dashid:
        croppedImage = newT[dbrecs[id][1]:dbrecs[id][1] + dbrecs[id][3], dbrecs[id][0]:dbrecs[id][0] + dbrecs[id][2]].copy()
        cv2.floodFill(croppedImage, None, (int(croppedImage.shape[1] / 2) - 1, int(croppedImage.shape[0] / 2) - 1), 255)
        if (np.average(croppedImage) > 0.6 * 255):
            dashes = True


    for id in contIds:

        allDiffs = []

        croppedImage = newT[brecs[id][1]:brecs[id][1] + brecs[id][3], brecs[id][0]:brecs[id][0] + brecs[id][2]]


        ccontours, chier = cv2.findContours(croppedImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        cropchecked = []
        cbrecs = {}
        ccontIds = []

        for ci in range(len(ccontours)):
            cbrec = cv2.boundingRect(ccontours[ci])
            if ([cbrec[0], cbrec[1]] not in cropchecked):
                cropchecked.append([cbrec[0], cbrec[1]])
                cbrecs[ci] = cbrec
                ccontIds.append(ci)

        cropchildMap = defaultdict(list)

        for cid in ccontIds[:]:
            isInside = check_inside(cbrecs, cbrecs[cid], cid)
            if isInside:
                # print("c")
                cropchildMap[isInside - 1].append(ccontours[cid])
                ccontIds.remove(cid)
                del cbrecs[cid]

        for cid in ccontIds:
            cv2.drawContours(croppedImage, ccontours, cid, 255, -1)
            cv2.drawContours(croppedImage, cropchildMap[cid], -1, 0, -1)

        width = int(croppedImage.shape[1])
        height = int(croppedImage.shape[0])



        croppedImage = cv2.morphologyEx(croppedImage, cv2.MORPH_OPEN, np.ones([int(np.ceil(width/20)), int(np.ceil(height/20))]), iterations=1)

        dim = (width, height)
        #print(dim)

        for t in range(0, 27):
            image = trIm[t]
            resized = cv2.resize(image, dim, interpolation=cv2.INTER_NEAREST)



            diff = np.logical_xor(resized, croppedImage, dtype=np.int16)

            diff = np.sum(diff) / (croppedImage.shape[0] * croppedImage.shape[1])

            allDiffs.append(diff)

            #cv2.namedWindow("t", cv2.WINDOW_NORMAL)
            #cv2.imshow("t", np.hstack([croppedImage,resized] ))
            #cv2.waitKey()
        minIndex = np.argmin(allDiffs)

        if (allDiffs[minIndex] < 0.35):

            brecVals.append(brecs[id][0])

            plate.append(minIndex)

    finalPlate = []


    valsLength = len(brecVals)

    sortedMin = np.argsort(brecVals)

    #cv2.namedWindow("t", cv2.WINDOW_NORMAL)
    #cv2.imshow("t", newT)
    #cv2.waitKey()

    for i in range(0, valsLength):
        finalPlate.append(plate[sortedMin[i]])
    if len(finalPlate) >= 6:
        #print(finalPlate)
        if dashes:
            finalPlate.append(27)

        return tuple(finalPlate)
    else:
        return []



def check_inside(br_list, check, br_id):
    for id, br in br_list.items():
        if check[0] >= br[0] and check[1] >= br[1] and check[0] + check[2] <= br[0] + br[2] and check[1] + check[3] <= \
                br[3] + br[3] and br_id != id:
            return id + 1
    return False

