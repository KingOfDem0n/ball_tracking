import numpy as np

class histogramFeature(object):
    def __init__(self, img):
        self.img = img.copy()
        self.size = img.shape[0] * img.shape[1]
        self.P_map = np.array([self.P(g) for g in range(255)]).reshape(-1)
        self.mean = self.histMean()
        self.std = self.histStd()
        self.skew = self.histSkew()
        self.energy = self.histEnergy()
        self.entropy = self.histEntropy()

    def P(self, g):
        return np.sum(self.img == g) / self.size

    def histMean(self):
        return np.sum(self.img) / self.size

    def histStd(self):
        return np.sqrt(np.sum(((np.arange(255)-self.mean)**2)*self.P_map)).reshape(-1)

    def histSkew(self):
        return np.sum(((np.arange(255) - self.mean) ** 3) * self.P_map)/(self.std**3)

    def histEnergy(self):
        return np.sum(self.P_map**2)

    def histEntropy(self):
        return -np.sum(self.P_map*np.log2(self.P_map))



