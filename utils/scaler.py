import numpy as np

class MyStandardScaler():
    def __init__(self):
        pass
    
    def fit(self, X):
        self.mean_ = np.mean(X)
        self.std_ = np.std(X)
        
    def transform(self, X):
        X_tr = np.copy(X)
        X_tr -= self.mean_
        X_tr /= self.std_
        return X_tr