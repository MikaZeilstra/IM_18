import threading
import Localization
import Recognize
import cv2
import collections
import sys

class disQueue:
    queue = []
    qlock = threading.Lock()
    rlock = threading.Lock()
    olock = threading.Lock()
    alive = False
    threads = []
    trIm = []
    platesList = collections.defaultdict(list)
    total = -1

    def __init__(self, total):
        for y in range(0, 10):
            trainingImage = cv2.imread('SameSizeNumbers/' + str(y) + '.bmp', cv2.IMREAD_GRAYSCALE)  # trainImage
            self.trIm.append(trainingImage)

        for y in range(1, 18):
            trainingImage = cv2.imread('SameSizeLetters/' + str(y) + '.bmp', cv2.IMREAD_GRAYSCALE)  # trainImage
            self.trIm.append(trainingImage)
        self.total = total

    def addFrame(self,frame):
        with self.qlock:
            self.queue.append(frame)

    def startWork(self):
        self.alive = True
        while(len(self.threads) <= 8):
            nThread = threading.Thread(target=self.work)
            nThread.start()
            self.threads.append(nThread)

    def work(self):
        while self.alive or self.queue:
            if(len(self.queue) != 0):
                with self.olock:
                    sys.stdout.write("\r"+ str(1-(len(self.queue)/self.total)))
                    sys.stdout.flush()
            jc = 0
            cplates = []
            cStamps = []
            cQueue = []
            with self.qlock:
                while(jc <= 10 and self.queue):
                    jc += 1
                    cQueue.append(self.queue.pop(0))
            while cQueue:
                cJ = cQueue.pop(0)

                plate = Localization.plate_detection(cJ[0])

                rec = []
                for im in plate:
                    rec = Recognize.segment_and_recognize(im, self.trIm)

                if len(rec) != 0:
                    cplates.append(rec)
                    cStamps.append(cJ[1])

            with self.rlock:
                for p,s in zip(cplates,cStamps):
                   self.platesList[p].append(s)


    def getResult(self):
        self.alive = False
        for t in self.threads:
            t.join()
        return self.platesList
