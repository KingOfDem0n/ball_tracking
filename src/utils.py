import cv2 as cv

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