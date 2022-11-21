import numpy as np

class Neighbors_Laplacian:
    """
    Each number M(i,j) in the matrix is similarity between points (i,j)
    """
    def __init__(self, Lp=None, nearest=None, algorithm="fair"):
        self.Lp = Lp
        self.nearest = nearest
        self.algorithm = algorithm

    def prune(self):
        assert self.Lp.shape == self.Lp.T.shape
        if self.algorithm.lower() == "fair":
            return self.neighbors_fair()
        else:
            raise Exception("!! NO SUCH ALGORITHM")

    def neighbors_fair(self):
        if self.nearest is None:
            return self.Lp
        nrow, ncol = self.Lp.shape
        keepN = min(self.nearest, nrow)
        newLp = np.zeros((nrow, ncol))
        for row in range(nrow):
            index = np.argsort(self.Lp[row])[::-1][:keepN]
            newLp[row][index] = self.Lp[row][index] 
        return np.maximum(newLp, newLp.T)


class Neighbors_DataPoint:
    """
    Each row is a datapoint
    """
    def __init__(self, dataMat=None, nearest=None, algorithm=None):
        self.dataMat = dataMat
        self.nearest = nearest
        self.algorithm = algorithm

