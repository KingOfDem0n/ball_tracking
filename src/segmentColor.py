#!/usr/bin/env python

import math
import numpy as np
import cv2 as cv

def findBox(contours, thresh=0.8):
    new_cnt = []

    for cnt in contours:
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cnt_area = cv.contourArea(cnt)
        box_area = cv.contourArea(box)
        if box_area > 0 and cnt_area > 70 and cnt_area/box_area >= thresh:
            new_cnt.append(cnt)

    return new_cnt

def approximateContour(contours, k):
    approx = []
    for cnt in contours:
        epsilon = k*cv.arcLength(cnt,True)
        approx.append(cv.approxPolyDP(cnt,epsilon,True))
    return approx

def roatedRectangle(contours):
    boxes = []
    for cnt in contours:
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        boxes.append(np.int0(box))
    return boxes

def checkSolidity(contours, thresh):
    solid = []
    for cnt in contours:
        hull = cv.convexHull(cnt)
        hull_area = cv.contourArea(hull)
        cnt_area = cv.contourArea(cnt)
        if hull_area > 0 and cnt_area/hull_area >= thresh:
            solid.append(cnt)
    return solid

def getCenter(contours):
    center = []
    for cnt in contours:
        M = cv.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        center.append((cx, cy))
    return tuple(center)

def drawCenters(frame, centers):
    for c in centers:
        cv.circle(frame, c, 3, (0,0,255), -1)

if __name__ == "__main__":
    cap = cv.VideoCapture(1)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask1 = cv.inRange(hsv, (0, 100, 100), (5, 255, 255))
        mask2 = cv.inRange(hsv, (175, 100, 100), (180, 255, 255))
        mask = cv.bitwise_or(mask1, mask2)

        result = cv.bitwise_and(frame, frame, mask=mask)
        gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
        _, result = cv.threshold(gray, 0, 255, 0)
        im2, contours, hierarchy = cv.findContours(result, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        new_contours = findBox(contours, 0.7)
        new_contours = checkSolidity(new_contours, 0.5)
        boxes = roatedRectangle(new_contours)

        centers = getCenter(new_contours)
        result = cv.cvtColor(result, cv.COLOR_GRAY2BGR)
        cv.drawContours(result, new_contours, -1, (0,255,0), 3)
        drawCenters(result, centers)

        # Display the resulting frame
        cv.imshow('frame',result)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
