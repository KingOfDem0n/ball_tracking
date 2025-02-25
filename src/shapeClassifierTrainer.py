#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime
from scipy.optimize import nnls
from shapefeatures import *
from utils import *
import cv2 as cv


def train(classes, save_mode=False):
    for c in classes:
        pic_names = os.listdir("../images/Train/{}".format(c))
        features = []
        for n in tqdm(pic_names, desc=c):
            img = cv.imread("../images/Train/{}/{}".format(c, n))
            processed = preprocess(img)
            contours, _ = cv.findContours(processed, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            for cnt in contours:
                if cv.contourArea(cnt) >= 300:
                    bgr = cv.cvtColor(processed, cv.COLOR_GRAY2BGR)
                    cv.drawContours(bgr, cnt, -1, (0, 255, 0), 2)
                    cv.imshow("Contours", bgr)
                    key = cv.waitKey(0) & 0xFF
                    if key == ord('s'):
                        features.append(customFeatures(cnt))
                        print("Save features!")
        if save_mode == 1:
            saveFeatures(c, features)
    cv.destroyAllWindows()

def test(target, classes, display=False):
    ref = loadReference("../featureInfo/normalizedShapeFeatures.txt", classes)
    norm_param = np.array(loadFeatures("../featureInfo/normalizedParameters.txt"))
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
                if cv.contourArea(cnt) >= 500:
                    feat = np.array(customFeatures(cnt))
                    norm = (feat - norm_param[0])/norm_param[1]
                    result = compare(one_ref, norm)
                    dist = result[1]
                    if display:
                        print("Euclid: {}".format(result[0]))
                        print("NVIP: {}".format(result[1]))
                        print("Tanimoto: {}".format(result[2]))
                        print("R2: {}".format(result[3]))
                        bgr = cv.cvtColor(processed, cv.COLOR_GRAY2BGR)
                        cv.drawContours(bgr, cnt, -1, (0, 255, 0), 2)
                        cv.imshow("Contours", bgr)
                        cv.waitKey(0)
                    if dist >= 0.95:
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

def getPredictAndTruth(classes):
    ref = loadReference("../featureInfo/normalizedShapeFeatures.txt", classes)
    norm_param = np.array(loadFeatures("../featureInfo/normalizedParameters.txt"))
    output = []
    for c in classes:
        one_ref = np.array(ref[c])
        pic_names = os.listdir("../images/Train/{}".format(c))
        for n in tqdm(pic_names, desc=c):
            img = cv.imread("../images/Train/{}/{}".format(c, n))
            processed = preprocess(img)
            contours, _ = cv.findContours(processed, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            for cnt in contours:
                if cv.contourArea(cnt) >= 300:
                    bgr = cv.cvtColor(processed, cv.COLOR_GRAY2BGR)
                    cv.drawContours(bgr, cnt, -1, (0, 255, 0), 2)
                    cv.imshow("Contours", bgr)
                    key = cv.waitKey(0) & 0xFF
                    feat = np.array(customFeatures(cnt))
                    norm = (feat - norm_param[0]) / norm_param[1]
                    if key == ord('s'):
                        output.append(list(compare(one_ref, norm)[1:]) + [1])
                    else:
                        output.append(list(compare(one_ref, norm)[1:]) + [-1])

    saveFeatures("PredictionAndTruth", output)

    cv.destroyAllWindows()

def findEnsembleCoeff():
    info = np.array(loadFeatures("../featureInfo/PredictionAndTruth.txt"))
    A = info[:, :-1]
    b = info[:, -1]

    out = nnls(A, b)

    return out

def testAll(classes, mode=0, save=True, display=False):
    result = []
    num_test = []
    ref = loadReference("../featureInfo/shapeFeatures.txt", classes)
    # result.append(test(c, classes, display))

    for c in classes:
        pic_names = os.listdir("../images/Test/{}".format(c))
        num_test.append(len(pic_names))

    for i, target in enumerate(tqdm(classes, desc="Testing all classes")):
        result.append([0]*len(classes))
        for j, c in enumerate(classes):
            pic_names = os.listdir("../images/Test/{}".format(c))
            for n in pic_names:
                img = cv.imread("../images/Test/{}/{}".format(c, n))
                dist, pred, cnt, processed = predict(img, classes, ref, mode)

                if mode == 0 and dist <= 0.1 and pred == target:
                    result[i][j] += 1
                elif dist > 0.9 and pred == target:
                    result[i][j] += 1

                if display and target == c:
                    print("Metric: {}".format(dist), end="\t")
                    print("Predict: {}".format(pred))
                    bgr = cv.cvtColor(processed, cv.COLOR_GRAY2BGR)
                    cv.drawContours(bgr, cnt, -1, (0, 255, 0), 2)
                    cv.imshow("Contours", bgr)
                    cv.waitKey(0)

    # output = np.vstack((result, num_test))
    output = np.array(result).T
    if save:
        df = pd.DataFrame(output, index=classes, columns=classes)

        filler = datetime.datetime.now()
        path = "../results/{}-{}-{}-{}-{}-Tanimoto.xlsx".format(filler.year, filler.month, filler.day, filler.hour, filler.minute)

        df.to_excel(path)

    return output

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

def normalizeAllFeatures(classes):
    features = []
    all_feat = []
    for c in classes:
        feat = loadFeatures("../featureInfo/{}.txt".format(c))
        mean, std = processFeatures(feat)
        features.append(mean)
        all_feat += feat

    mean, std = processFeatures(all_feat)
    norm_param = np.vstack((mean, std)).tolist()
    features = np.array(features)
    norm = ((features - mean)/std).tolist()
    saveFeatures("normalizedParameters", norm_param)
    saveFeatures("normalizedShapeFeatures", norm)

if __name__ == "__main__":
    classes = ["5-points-Star", "8-points-Star", "Arrow", "Heart", "Octagon", "Rainbow", "Triangle"]
    result = testAll(classes, 0, False, True)
    print(result)

# Eculidean distance seems to be around 0.1 when target is selected

