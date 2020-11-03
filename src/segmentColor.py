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

def checkSize(contours, thresh=70):
    new_cnt = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area >= thresh:
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

def segmentRedBox(frame):
    hsv = cv.cvtColor(frame.copy(), cv.COLOR_BGR2HSV)
    mask1 = cv.inRange(hsv, (0, 70, 70), (8, 255, 255))
    mask2 = cv.inRange(hsv, (170, 70, 70), (180, 255, 255))
    mask = cv.bitwise_or(mask1, mask2)

    result = cv.bitwise_and(frame, frame, mask=mask)
    result = cv.medianBlur(result, 21)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
    result = cv.morphologyEx(result, cv.MORPH_OPEN, kernel)
    gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
    _, result = cv.threshold(gray, 0, 255, 0)
    im2, contours, hierarchy = cv.findContours(result, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    new_contours = findBox(contours, 0.7)
    new_contours = checkSolidity(new_contours, 0.5)
    new_contours = checkSize(new_contours, 200)

    return new_contours

def narrowSearchSpace(frame, contours):
    boxes = roatedRectangle(contours)
    e = 50
    enlarged_boxes = []
    for box in boxes:
        x,y,w,h = cv.boundingRect(box)
        cv.rectangle(frame,(x-e,y-e),(x+w+e,y+h+e),(0,0,255),2)

if __name__ == "__main__":
    cap = cv.VideoCapture(1)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        new_contours = segmentRedBox(frame)
        narrowSearchSpace(frame, new_contours)

        centers = getCenter(new_contours)
        # result = cv.cvtColor(result, cv.COLOR_GRAY2BGR)
        cv.drawContours(frame, new_contours, -1, (0,255,0), 3)
        drawCenters(frame, centers)

        # Display the resulting frame
        cv.imshow('frame',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
