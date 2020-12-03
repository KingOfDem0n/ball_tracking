import numpy as np
import cv2 as cv

def circlarityRatio(cnt):
    P = cv.arcLength(cnt, True)
    A = cv.contourArea(cnt)

    return 4*np.pi*A/(P**2)

def rectanglarity(cnt):
    rect = cv.minAreaRect(cnt)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    cnt_area = cv.contourArea(cnt)
    box_area = cv.contourArea(box)

    return cnt_area/box_area

def convexity(cnt):
    hull = cv.convexHull(cnt)
    P = cv.arcLength(cnt, True)
    hull_P = cv.arcLength(hull, True)

    return hull_P/P

def solidity(cnt):
    hull = cv.convexHull(cnt)
    hull_area = cv.contourArea(hull)
    cnt_area = cv.contourArea(cnt)

    return cnt_area/hull_area

def customFeatures(cnt):
    c = circlarityRatio(cnt)
    r = rectanglarity(cnt)
    C = convexity(cnt)
    s = solidity(cnt)
    m = cv.moments(cnt)
    hu = cv.HuMoments(m).flatten()

    return [c, r, C, s] + hu.tolist()
