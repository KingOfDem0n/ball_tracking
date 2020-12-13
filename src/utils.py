import cv2 as cv
import numpy as np
from shapefeatures import customFeatures

def preprocess(img):
    _, binary = cv.threshold(img, 127, 255, type=cv.THRESH_BINARY)

    hsv = cv.cvtColor(binary, cv.COLOR_BGR2HSV)
    thresh = cv.inRange(hsv, (0, 0, 0), (180, 255, 80))
    smooth = cv.medianBlur(thresh, 5)
    return smooth

def saveFeatures(c, features):
    with open("../featureInfo/{}.txt".format(c), 'w') as f:
        for feat in features:
            f.write(str(feat)[1:-1] + "\n")

def loadFeatures(file):
    features = []
    with open(file, "r") as f:
        line = f.readline()
        while line:
            elements = line.split(", ")
            feat = list(map(float, elements))
            features.append(feat)
            line = f.readline()
    return features

def loadReference(file, classes):
    ref = dict()
    count = 0
    with open(file, "r") as f:
        line = f.readline()
        while line:
            elements = line.split(", ")
            feat = list(map(float, elements))
            ref[classes[count]] = feat
            count += 1
            line = f.readline()
    return ref

def euclideanDist(A, B):
    return np.sqrt(np.sum((A - B)**2))

def NVIP(A, B):
    denom = np.sum(A*B)
    numer = np.sqrt(np.sum(A**2))*np.sqrt(np.sum(B**2))

    return denom/numer

def Tanimoto(A, B):
    ab = np.sum(A*B)
    a2 = np.sum(A**2)
    b2 = np.sum(B**2)

    return ab/(a2 + b2 - ab)

def R2(A, B):
    a_bar = np.mean(A)
    b_bar = np.mean(B)
    denom = np.sum((A-a_bar)*(B-b_bar))
    numer = np.sqrt(np.sum((A-a_bar)**2)*np.sum((B-b_bar)**2))

    return denom/numer

def compare(ref, feat):
    euclid = euclideanDist(ref, feat)
    nvip = NVIP(ref, feat)
    T = Tanimoto(ref, feat)
    r2 = R2(ref, feat)
    return euclid, nvip, T, r2

def predict(img, classes, ref, mode=0):
    processed = preprocess(img)
    contours, _ = cv.findContours(processed, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    min_dist = float('inf')
    max_similarity = -float('inf')
    contour = None
    pred = ""
    for cnt in contours:
        if cv.contourArea(cnt) >= 500:
            for c in classes:
                one_ref = np.array(ref[c])
                feat = np.array(customFeatures(cnt))
                dist = compare(one_ref, feat)[mode]
                if mode == 0 and dist < min_dist:
                    contour = cnt
                    min_dist = dist
                    pred = c
                elif dist > max_similarity:
                    contour = cnt
                    max_similarity = dist
                    pred = c

    if mode == 0:
        return min_dist, pred, contour, processed

    return max_similarity, pred, contour, processed
