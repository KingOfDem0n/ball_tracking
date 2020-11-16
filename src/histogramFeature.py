import numpy as np

class histogramFeature(object):
    def __init__(self, img, max_val=255, idx=None):
        self.img = img.copy()
        self.max_val = max_val
        self.idx = idx
        self.size = img.shape[0] * img.shape[1]
        if idx is not None:
            self.size = np.count_nonzero(idx)
        self.P_map = np.array([self.P(g) for g in range(self.max_val)]).reshape(-1)
        self.mean = self.histMean()
        self.std = self.histStd()
        self.skew = self.histSkew()
        self.energy = self.histEnergy()
        self.entropy = self.histEntropy()

    def P(self, g):
        if self.idx is None:
            return np.sum(self.img == g) / self.size
        else:
            return np.sum(self.img[self.idx] == g) / self.size

    def histMean(self):
        if self.idx is None:
            return np.sum(self.img) / self.size
        else:
            return np.sum(self.img[self.idx]) / self.size

    def histStd(self):
        return np.sqrt(np.sum(((np.arange(self.max_val)-self.mean)**2)*self.P_map)).reshape(-1)

    def histSkew(self):
        return np.nan_to_num(np.sum(((np.arange(self.max_val) - self.mean) ** 3) * self.P_map)/(self.std**3))

    def histEnergy(self):
        return np.sum(self.P_map**2)

    def histEntropy(self):
        return -np.sum(np.nan_to_num(self.P_map*np.log2(self.P_map)))

    def getFeatures(self):
        return [self.mean, self.std[0], self.skew[0], self.energy, self.entropy]


