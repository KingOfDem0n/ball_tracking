#!/usr/bin/env python

import os
import numpy as np
from shapefeatures import *
from features import getCenter, drawCenters
from utils import *
import cv2 as cv

if __name__ == "__main__":
    cap = cv.VideoCapture(1)
    classes = ["5-points-Star", "8-points-Star", "Arrow", "Heart", "Octagon", "Rainbow", "Triangle"]
    ref_norm = loadReference("../featureInfo/normalizedShapeFeatures.txt", classes)
    ref = loadReference("../featureInfo/shapeFeatures.txt", classes)
    norm_param = np.array(loadFeatures("../featureInfo/normalizedParameters.txt"))
    target = None #classes[0]

    try:
        while True:
            ret, frame = cap.read()

            dist, pred, cnt, debug = predict(frame, classes, ref, 0)
            bgr = cv.cvtColor(debug, cv.COLOR_GRAY2BGR)
            if cnt is not None and dist <= 0.1:
                if target is None:
                    center = getCenter([cnt])
                    drawCenters(frame, center)
                    drawCenters(bgr, center)
                    p_x = center[0][0] - frame.shape[1]//2
                    p_y = frame.shape[0]//2 - center[0][1]
                    shift_center = list(center[0])
                    shift_center[1] += 40
                    cv.drawContours(frame, cnt, -1, (0, 255, 0), 3)
                    cv.putText(frame, pred, center[0], cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
                    cv.putText(frame, str((p_x, p_y)), tuple(shift_center), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
                    cv.drawContours(bgr, cnt, -1, (0, 255, 0), 3)
                elif pred == target:
                    center = getCenter([cnt])
                    drawCenters(frame, center)
                    drawCenters(bgr, center)
                    p_x = center[0][0] - frame.shape[1]//2
                    p_y = frame.shape[0]//2 - center[0][1]
                    shift_center = list(center[0])
                    shift_center[1] += 40
                    cv.drawContours(frame, cnt, -1, (0, 255, 0), 3)
                    cv.putText(frame, target, center[0], cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
                    cv.putText(frame, str((p_x, p_y)), tuple(shift_center), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
                    cv.drawContours(bgr, cnt, -1, (0, 255, 0), 3)

            # Display the resulting frame
            cv.imshow('Frame/Debug', np.hstack((frame, bgr)))

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()
