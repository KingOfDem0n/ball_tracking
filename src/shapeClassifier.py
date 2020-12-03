#!/usr/bin/env python

import os
import numpy as np
from shapefeatures import *
from utils import *
import cv2 as cv

if __name__ == "__main__":
    cap = cv.VideoCapture(1)

    ref = loadReference("../featureInfo/shapeFeatures.txt")
    classes = ["5-points-Star", "8-points-Star", "Arrow", "Heart", "Octagon", "Rainbow", "Triangle"]

    try:
        while True:
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

    finally:
        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()