import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
import torch
import torch.nn as nn
from tqdm import tqdm

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
        # meta_indices = permutation[int(len(X) * self.train_rate):]
        meta_indices = permutation

        self.regressor.fit(X[train_indices], y[train_indices])
        self.meta_regressor.fit(self._transform(X[meta_indices]), y[meta_indices])

        return self
    
    def predict(self, X):
        return self.meta_regressor.predict(self._transform(X))

# technically it is trained as a classifier, but it is used as a regressor
# it is trained to model whether the label will be positive or negative
class ForwardForwardRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, output_size, threshold=1, epochs=100, batch_size=256):
        self.output_size = output_size
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.threshold = threshold
        self.epsilon = 1e-6
        self.batch_size = batch_size
    
    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32)
        input_size = X.shape[1]
        y = torch.tensor(1 * (y > 0), dtype=torch.float32)
        joined = torch.cat([X, y.unsqueeze(1)], dim=1)
        joined_negatives = torch.cat([X, (1 - y).unsqueeze(1)], dim=1)
        self.model = nn.Sequential(
            nn.Linear(input_size + 1, self.output_size),
            nn.ReLU()
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        for _ in tqdm(range(self.epochs)):
            for batch_start in range(0, len(X), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(X))
                optimizer.zero_grad()

                positive_output = self.model(joined[batch_start:batch_end].to(self.device))
                positive_goodness = torch.mean(positive_output ** 2, dim=1) - self.threshold
                positive_probability = torch.sigmoid(positive_goodness)

                negative_output = self.model(joined_negatives[batch_start:batch_end].to(self.device))
                negative_goodness = torch.mean(negative_output ** 2, dim=1) - self.threshold
                negative_probability = torch.sigmoid(negative_goodness)
                
                loss = -torch.mean(torch.log(positive_probability + self.epsilon) + torch.log(1 - negative_probability + self.epsilon))
                loss.backward()
                optimizer.step()
        
        return self

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        joined = torch.cat([X, torch.ones(X.shape[0], 1, device=self.device)], dim=1)
        joined_negatives = torch.cat([X, torch.zeros(X.shape[0], 1, device=self.device)], dim=1)
        with torch.no_grad():
            positive_output = self.model(joined)
            positive_goodness = torch.mean(positive_output ** 2, dim=1) - self.threshold

            negative_output = self.model(joined_negatives)
            negative_goodness = torch.mean(negative_output ** 2, dim=1) - self.threshold
            
            return (positive_goodness - negative_goodness).cpu().numpy()
