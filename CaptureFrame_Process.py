import cv2
import numpy as np
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
    cap = cv2.VideoCapture(file_path)
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO,1)
    t_total = cap.get(cv2.CAP_PROP_POS_MSEC)
    f_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
    spf = int(np.round(t_total/ f_total))
    print(spf)
    while(True):
        ret, frame = cap.read()
        #print(ret)
        if(not(ret)):
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
            continue

        start = time.time();
        plate = Localization.plate_detection(frame)
        print(time.time() - start)
        #print(plate.shape)
        #for im in plate:
            #Recognize.segment_and_recognize(im)
            #pass
        cv2.imshow('frame', plate)
        if cv2.waitKey(spf) & 0xFF == ord('q'):
               break
    #cv2.imwrite("BinTemplate.jpg", t)

    cv2.destroyAllWindows()
    cap.release()