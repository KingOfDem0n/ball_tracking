#!/usr/bin/env python

import os
import numpy as np
from shapefeatures import *
from features import getCenter, drawCenters
from utils import *
import cv2 as cv

def predict(img, classes, ref, ref_norm, norm_param):
    processed = preprocess(img)
    contours, _ = cv.findContours(processed, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    max_dist = -1
    min_dist = float('inf')
    contour = None
    pred = ""
    for cnt in contours:
        if cv.contourArea(cnt) >= 300:
            for c in classes:
                one_ref = np.array(ref[c])
                feat = np.array(customFeatures(cnt))
                dist = compare(one_ref, feat)[0]
                if dist < min_dist:
                    contour = cnt
                    min_dist = dist
                    pred = c

    return min_dist, pred, contour, processed

if __name__ == "__main__":
    cap = cv.VideoCapture(1)
    classes = ["5-points-Star", "8-points-Star", "Arrow", "Heart", "Octagon", "Rainbow", "Triangle"]
    ref_norm = loadReference("../featureInfo/normalizedShapeFeatures.txt", classes)
    ref = loadReference("../featureInfo/shapeFeatures.txt", classes)
    norm_param = np.array(loadFeatures("../featureInfo/normalizedParameters.txt"))
    target = classes[0]

    try:
        while True:
            ret, frame = cap.read()

            dist, pred, cnt, debug = predict(frame, classes, ref, ref_norm, norm_param)
            bgr = cv.cvtColor(debug, cv.COLOR_GRAY2BGR)
            if cnt is not None and dist <= 0.1:
                if target is None:
                    center = getCenter([cnt])
                    drawCenters(frame, center)
                    drawCenters(bgr, center)
                    cv.drawContours(frame, cnt, -1, (0, 255, 0), 3)
                    cv.drawContours(bgr, cnt, -1, (0, 255, 0), 3)
                elif pred == target:
                    center = getCenter([cnt])
                    drawCenters(frame, center)
                    drawCenters(bgr, center)
                    cv.drawContours(frame, cnt, -1, (0, 255, 0), 3)
                    cv.drawContours(bgr, cnt, -1, (0, 255, 0), 3)

            # Display the resulting frame
            cv.imshow('Frame/Debug', np.hstack((frame, bgr)))

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()