import numpy as np
import cv2 as cv
from histogramFeature import histogramFeature

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
    mask1 = cv.inRange(hsv, (0, 50, 50), (10, 255, 255)) # (0, 70, 70), (8, 255, 255)
    mask2 = cv.inRange(hsv, (170, 50, 50), (180, 255, 255)) # (170, 70, 70), (180, 255, 255)
    mask = cv.bitwise_or(mask1, mask2)

    result = cv.bitwise_and(frame, frame, mask=mask)
    result = cv.medianBlur(result, 21)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
    result = cv.morphologyEx(result, cv.MORPH_OPEN, kernel)
    gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
    _, result = cv.threshold(gray, 0, 255, 0)
    contours, hierarchy = cv.findContours(result, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    new_contours = checkSolidity(contours, 0.5)
    new_contours = findBox(new_contours, 0.7)
    new_contours = checkSize(new_contours, 200)

    return new_contours

def narrowSearchSpace(contours):
    e = 0
    boxes = []
    for cnt in contours:
        x,y,w,h = cv.boundingRect(cnt)
        boxes.append([(x-e,y-e),(x+w+e,y+h+e)])
        # cv.rectangle(frame,(x-e,y-e),(x+w+e,y+h+e),(0,0,255),2)
    return boxes

def costumFeatures(frame, contours, onlyLargest=False):
    """
    Combination of laws texture histogram and laws texture features
    :param frame: Image to process
    :param contours: Contour of objects of interests
    :param onlyLargest: To only use the the largest area contour
    :return:
        feat: list(25) A list of features in the following orders [Histogram, Laws LS, Laws SW, Laws SS, Laws RR]
        useCnt: None|contour If onlyLargest is true, then return the largest contour used
    """
    vector = {"L": np.array([1, 4, 6, 4, 1]),
              "E": np.array([-1, -2, 0, 2, 1]),
              "S": np.array([-1, 0, 2, 0, -1]),
              "R": np.array([1, -4, 6, -4, 1]),
              "W": np.array([-1, 2, 0, -2, 1])}
    feat = []

    # combination = ["LE", "LS", "LR", "LW", "EE", "ES", "ER", "EW", "SS", "SR", "SW", "RR", "RW", "WW"]

    combination = ["LS", "SW", "SS", "RR"]
    # Note:
    #   Not useful: LL

    red = frame[:, :, 2]
    maxArea = -1
    useCnt = None
    for cnt in contours:
        area = cv.contourArea(cnt)
        temp = []
        if area > maxArea:
            mask = np.zeros(red.shape[:2])
            mask = cv.fillPoly(mask, pts=[cnt], color=(255, 255, 255))
            mask = mask.astype(np.uint8)

            focused = cv.bitwise_and(red, red, mask=mask)
            idx = mask > 0

            avg = cv.blur(focused, (15, 15))
            filter_img = np.abs(focused - avg)

            # cv.imshow('delight', filter_img)

            histFeat = histogramFeature(filter_img, idx=idx)
            temp += histFeat.getFeatures()
            for c in combination:
                kernel = np.outer(vector[c[0]], vector[c[1]])
                texture = cv.filter2D(filter_img, -1, kernel)

                # cv.imshow('Texture {}'.format(c), texture)

                textureEnergyMap = cv.filter2D(np.abs(texture), -1, np.ones((3, 3)))
                # cv.imshow('Texture Energy Map {}'.format(c), textureEnergyMap)

                histFeat = histogramFeature(textureEnergyMap, idx=idx)
                temp += histFeat.getFeatures()

            if onlyLargest:
                feat.clear()
                feat.append(temp)
                maxArea = area
                useCnt = cnt
            else:
                feat.append(temp)

    return feat, useCnt
