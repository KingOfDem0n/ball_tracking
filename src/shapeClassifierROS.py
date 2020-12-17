#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Pose2D
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
    target = "5-points-Star"

    rospy.init_node('shapeClassifier', anonymous=True)
    pub = rospy.Publisher('/shapeCenter', Pose2D, queue_size=1)

    try:
        while True:
            ret, frame = cap.read()
            pose = Pose2D()

            dist, pred, cnt, debug = predict(frame, classes, ref, 0)
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
                    pose.x = center[0][0] - frame.shape[1]//2
                    pose.y = frame.shape[0]//2 - center[0][1]
                    print(pose.x, pose.y)
                    pub.publish(pose)

            # Display the resulting frame
            cv.imshow('Frame/Debug', np.hstack((frame, bgr)))

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()
