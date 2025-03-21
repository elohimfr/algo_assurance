import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, precision_recall_curve, roc_auc_score, roc_curve


# MLP

## train data
class TrainData(Dataset):
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __getitem__(self, index):
        return self.X[index], self.y[index]
        
    def __len__ (self):
        return len(self.X)

## test data    
class TestData(Dataset):
    
    def __init__(self, X):
        self.X = X
        
    def __getitem__(self, index):
        return self.X[index]
        
    def __len__ (self):
        return len(self.X)

def train_mlp(model, X_train: pd.DataFrame, y_train: pd.DataFrame, batch_size: int=64):
    # Train DataLoader
    train_data = TrainData(torch.tensor(X_train.values, dtype=torch.float32), 
               torch.tensor(y_train.values, dtype=torch.float32))
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    print(f'Train loader size: {len(train_loader)} batches')

    # Train model
    model.fit(train_loader)

def test_mlp(model, X_test, y_test, batch_size: int=64):
    # Test DataLoader
    test_data = TestData(torch.tensor(X_test.values, dtype=torch.float32))
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    print(f'Test loader size: {len(test_loader)} batches')
    
    # Model predictions
    pred = model.predict(test_loader)
    # Prediction probabilities
    pred_prob = model.predict_proba(test_loader)

    return pred, pred_prob



