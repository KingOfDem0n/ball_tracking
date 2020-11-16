#!/usr/bin/env python

import math
import os
import numpy as np
import cv2 as cv
from features import *
from scipy.optimize import nnls, lsq_linear
from tqdm import tqdm

def normalize(feat, stat):
    norm = np.zeros(feat.shape)
    for i in range(feat.shape[0]):
        norm[i] = (feat[i] - stat[i][0])/stat[i][1]
    return norm

def match(norm, ref):
    diff = (norm - ref).reshape((-1, 5))
    diff_sum = np.sum(diff**2, axis=1)
    diff_sqrt = np.sqrt(diff_sum)

    return diff_sqrt

def testImages(filename, savelog=False):
    log = []
    names = os.listdir(os.path.join(os.getcwd(), "..", "images", "Target"))
    for name in tqdm(names):
        frame = cv.imread(os.path.join(os.getcwd(), "..", "images", "Target", name))
        new_contours = segmentRedBox(frame)
        feat, cnt = costumFeatures(frame, new_contours, True)
        log.append(feat)

        cv.drawContours(frame, cnt, -1, (0, 255, 0), 3)

        # Display the resulting frame
        cv.imshow('frame', frame)
        cv.waitKey(1)

    print("Finished!")
    if savelog:
        np.save(os.path.join(os.getcwd(), "..", "featureInfo", filename), np.array(log))
    return np.array(log)

def parseLog(filename, savestat=False):
    array = np.load(os.path.join(os.getcwd(), "..", "featureInfo", filename + ".npy"))
    new_array = []
    for a in array:
        if len(a) > 0:
            new_array.append(a[0])
    new_array = np.array(new_array)
    mean = np.mean(new_array, axis=0)
    std = np.std(new_array, axis=0)

    if savestat:
        np.savetxt(os.path.join(os.getcwd(), "..", "featureInfo", filename + ".txt"), np.vstack((mean, std)).T)

    return np.vstack((mean, std)).T

def getNewFeatures(filename):
    testImages(filename, savelog=True)
    cv.destroyAllWindows()

    parseLog(filename + ".npy", True)

def runTest(testName, case="Others", display=True):
    names = os.listdir(os.path.join(os.getcwd(), "..", "images", case))
    stat = np.loadtxt(os.path.join(os.getcwd(), "..", "featureInfo", testName + ".txt"))
    ref = stat[:, 0]
    result = []
    for name in tqdm(names, desc="{} Case".format(case)):
        frame = cv.imread(os.path.join(os.getcwd(), "..", "images", case, name))
        new_contours = segmentRedBox(frame)
        feat, _ = costumFeatures(frame, new_contours, True)
        feat = np.array(feat)

        for j in range(feat.shape[0]):
            if feat[j].shape[0] > 0:
                diff = match(feat[j], ref)
                # diff[0] = 3 * diff[0]  # Histogram
                # diff[1] = 3 * diff[1]  # LS
                # diff[2] = 1.3 * diff[2]  # SW
                # diff[3] = 1.3 * diff[3]  # SS
                # diff[4] = 1.3 * diff[4]  # RR
                # Weighted coefficient: 3, 3, 1.3, 1.3, 1.3 with threshold of 40
                dist = np.sqrt(np.sum(diff ** 2))
                result.append(np.hstack((diff, dist)))
                if display:
                    print("\nIndividual: {}".format(diff))
                    print("\nAll: {}".format(dist))
            else:
                print("\nNo features")

            if display:
                cv.drawContours(frame, new_contours[j], -1, (0, 255, 0), 3)

                # Display the resulting frame
                cv.imshow('frame', frame)
                cv.waitKey(0)

    return result

def runFullTest(testName, save=True):
    targetResult = runTest(testName, "Target", False)
    othersResult = runTest(testName, "Others", False)
    if save:
        np.savetxt(os.path.join(os.getcwd(), "targetResult.txt"), np.array(targetResult))
        np.savetxt(os.path.join(os.getcwd(), "othersResult.txt"), np.array(othersResult))

def findCoefficient():
    # From target folder
    target_A = np.loadtxt(os.path.join(os.getcwd(), "targetResult.txt"))
    target_b = np.zeros(target_A.shape[0])
    # From others folder
    others_A = np.loadtxt(os.path.join(os.getcwd(), "othersResult.txt"))
    others_b = np.zeros(others_A.shape[0]) + 1000

    # Combined them together
    A = np.vstack((target_A, others_A))
    b = np.concatenate((target_b, others_b))

    out0 = nnls(A, b)
    out1 = lsq_linear(A, b, method="bvls", lsq_solver="exact", max_iter=3*A.shape[0])
    out = [[out0], [out1.x, out1.cost]]

    return out

if __name__ == "__main__":
    # runFullTest("targetFeature2")

    print(findCoefficient())


# Fail cases:
#   Bar code on the note book