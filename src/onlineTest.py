#!/usr/bin/env python

import os
import numpy as np
from features import *
import cv2 as cv

def match(norm, ref):
    diff = (norm - ref).reshape((-1, 5))
    diff_sum = np.sum(diff**2, axis=1)
    diff_sqrt = np.sqrt(diff_sum)

    return diff_sqrt

def compared(feat, contours, ref, threshold=40):
    chosen = []
    for f, cnt in zip(feat, contours):
        diff = match(f, ref)
        # print(f - ref)
        # diff = f - ref
        diff[0] = 3 * diff[0]  # Histogram
        diff[1] = 3 * diff[1]  # LS
        diff[2] = 1.3 * diff[2]  # SW
        diff[3] = 1.3 * diff[3]  # SS
        diff[4] = 1.3 * diff[4]  # RR

        dist = np.sqrt(np.sum(diff ** 2))
        print(diff)
        print(dist)

        if dist < threshold:
            chosen.append(cnt)

    return chosen

if __name__ == "__main__":
    cap = cv.VideoCapture(1)

    stat = np.loadtxt(os.path.join(os.getcwd(), "..", "featureInfo", "targetRedBandLawAndHistStat.txt"))
    ref = stat[:, 0]

    while(True):
        ret, frame = cap.read()


        new_contours = segmentRedBox(frame)
        feat, _ = costumFeatures(frame, new_contours, False)
        feat = np.array(feat)
        targetContour = compared(feat, new_contours, ref, 300)

        cv.drawContours(frame, targetContour, -1, (0, 255, 0), 3)

        # Display the resulting frame
        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

