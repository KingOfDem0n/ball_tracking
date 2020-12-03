#!/usr/bin/env python

import os
import numpy as np
from tqdm import tqdm
from shapefeatures import *
from utils import *
import cv2 as cv

def train(classes):
    for c in classes:
        pic_names = os.listdir("../images/Train/{}".format(c))
        features = []
        for n in tqdm(pic_names, desc=c):
            img = cv.imread("../images/Train/{}/{}".format(c, n))
            processed = preprocess(img)
            contours, _ = cv.findContours(processed, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            for cnt in contours:
                if cv.contourArea(cnt) >= 100:
                    bgr = cv.cvtColor(processed, cv.COLOR_GRAY2BGR)
                    cv.drawContours(bgr, cnt, -1, (0, 255, 0), 2)
                    cv.imshow("Contours", bgr)
                    key = cv.waitKey(0) & 0xFF
                    if key == ord('s'):
                        features.append(customFeatures(cnt))
                        print("Save features!")
        saveFeatures(c, features)
    cv.destroyAllWindows()

def test(target, classes, display=False):
    ref = loadReference("../featureInfo/shapeFeatures.txt", classes)
    one_ref = np.array(ref[target])
    correct = 0
    incorrect = 0
    num_test = 0
    for c in classes:
        pic_names = os.listdir("../images/Test/{}".format(c))
        if target == c:
            num_test = len(pic_names)
        for n in pic_names:
            img = cv.imread("../images/Test/{}/{}".format(c, n))
            processed = preprocess(img)
            contours, _ = cv.findContours(processed, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            for cnt in contours:
                if cv.contourArea(cnt) >= 300:
                    feat = np.array(customFeatures(cnt))
                    dist = np.sqrt(np.sum((one_ref - feat)**2))
                    if display:
                        print("Distance: {}".format(dist))
                        bgr = cv.cvtColor(processed, cv.COLOR_GRAY2BGR)
                        cv.drawContours(bgr, cnt, -1, (0, 255, 0), 2)
                        cv.imshow("Contours", bgr)
                        cv.waitKey(0)
                    if dist < 0.1:
                        if target == c:
                            if display:
                                print("Correct!")
                            correct += 1
                        else:
                            if display:
                                print("Incorrect")
                            incorrect += 1
                        if display:
                            print("Target spotted!")
                            print("Target: {}".format(target))
                            print("Classification: {}".format(c))
                            cv.waitKey(0)

    return correct/num_test, incorrect

def testAll(classes, display=False):
    result = []
    for c in tqdm(classes, desc="Testing all classes"):
        result.append(test(c, classes, display))

    return result

def processFeatures(feat):
    array = np.array(feat)
    mean = np.mean(array, axis=0).flatten()
    std = np.std(array, axis=0).flatten()

    return mean, std

def processAllFeatures(classes):
    features = []
    for c in classes:
        feat = loadFeatures("../featureInfo/{}.txt".format(c))
        mean, std = processFeatures(feat)
        features.append(mean.tolist())
    saveFeatures("shapeFeatures", features)

if __name__ == "__main__":
    cap = cv.VideoCapture(1)
    classes = ["5-points-Star", "8-points-Star", "Arrow", "Heart", "Octagon", "Rainbow", "Triangle"]
    # test("Heart", classes, True)
    results = testAll(classes, display=False)
    print(results)


