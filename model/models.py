import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class MetaRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, regressor, meta_regressor, train_rate=0.5):
        self.regressor = regressor
        self.meta_regressor = meta_regressor
        self.train_rate = train_rate

    def _transform(self, X):
        predictions = self.regressor.predict(X)

        return np.concatenate([
            X, predictions.reshape(-1, 1) # .repeat(X.shape[1], axis=1)
        ], axis=1)
        
    def fit(self, X, y):
        permutation = np.random.permutation(len(X))
        train_indices = permutation[:int(len(X) * self.train_rate)]
        meta_indices = permutation[int(len(X) * self.train_rate):]

        self.regressor.fit(X[train_indices], y[train_indices])
        self.meta_regressor.fit(self._transform(X[meta_indices]), y[meta_indices])

        return self
    
    def predict(self, X):
        return self.meta_regressor.predict(self._transform(X))
