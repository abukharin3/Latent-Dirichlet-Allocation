import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class pca:
    def __init__(self, data):
        self.data = data
        self.pca = self.get_pca()

    def get_pca(self):
        X = np.matmul(self.data.T, self.data)
        pca = PCA() # Found empirically
        pca.fit(X)
        return pca

    def show(self):
        print(self.pca.components_)

    def save(self):
        data = self.pca.transform(self.data)
        print(np.shape(data))






# Load data
train_path = r"data\train.npy"
train = np.load(train_path, allow_pickle = True)[0:50]

pc = pca(train)
pc.show()
