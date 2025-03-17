import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torcheval.metrics.classification import BinaryRecall, BinaryPrecision, BinaryF1Score, BinaryAccuracy
from fraudmrm.utils import TrainData, TestData


class BinaryMLP1L(nn.Module):
    def __init__(self, 
                 input_size: int,
                 layer1_size: int=128,
                 learning_rate: float=1e-3,
                 max_epochs: int=200,
                 batch_size: int=64,
                 tol: float=1e-4,
                 n_consec_iter: int=10,
                 verbose: bool=True,
                 random_state: int=None,
                ) -> None:
        super().__init__()

        # Set random seed for reproducibility
        if random_state:
            self.set_random_seed(random_state)
        
        # Training parameters
        self.input_size = input_size
        self.layer1_size = layer1_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.tol = tol
        self.n_consec_iter = n_consec_iter
        self.verbose = verbose

        # Layers
        self.layer1 = nn.Linear(self.input_size, self.layer1_size)
        self.layer_out = nn.Linear(self.layer1_size, 1)
        # Activation, Dropout, and Batch Normalisation
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.batchnorm1 = nn.BatchNorm1d(self.layer1_size)

    def set_random_seed(self, seed: int):
        """
        Sets the seed for reproducibility.
        """
        # Set the random seed for Python's random module, NumPy, and PyTorch
        # random.seed(seed) # Uncomment if using module random
        np.random.seed(seed)
        torch.manual_seed(seed)  # CPU
        torch.cuda.manual_seed(seed)  # GPU (if available)
        torch.cuda.manual_seed_all(seed)  # All GPUs
        # torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
        # torch.backends.cudnn.benchmark = False  # Disable the auto-tuner for deterministic behavior

    def forward(self, x):
        # Layer 1
        x = self.relu(self.layer1(x))
        # x = self.batchnorm1(x)
        x = self.dropout(x)
        # Output layer
        x = self.layer_out(x)

        return x

    # Modified version of .fit() for pd.DataFrame
    # def fit(self, train_loader: DataLoader):
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        # Send model to device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(device)

        # Train DataLoader
        train_data = TrainData(torch.tensor(X_train.values, dtype=torch.float32), 
                   torch.tensor(y_train.values, dtype=torch.float32))
        train_loader = DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True)
        if self.verbose: print(f'Train loader size: {len(train_loader)} batches')

        # Loss and Optimizer
        loss_function = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # Stop criterion variables
        avg_loss_old, count_iter = float('inf'), 0

        # Initialize metrics
        recall = BinaryRecall().to(device)
        precision = BinaryPrecision().to(device)
        f1 = BinaryF1Score().to(device)
        accuracy = BinaryAccuracy().to(device)

        # Training loop
        self.train()
        for e in range(1, self.max_epochs+1):
            epoch_loss = 0
            
            # Reset epoch loss and metrics
            recall.reset()
            precision.reset()
            f1.reset()
            accuracy.reset()
        
            for X_batch, y_batch in train_loader:
                # Send data to device
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                y_logit = self(X_batch)
                y_pred_prob = torch.sigmoid(y_logit)
                y_pred = torch.round(y_pred_prob)

                # Compute loss
                loss = loss_function(y_logit, y_batch.unsqueeze(1))#.view(-1, 1))
                epoch_loss += loss.item()
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                # Compute metrics
                recall.update(y_pred_prob.view(-1), y_batch.int())
                precision.update(y_pred_prob.view(-1), y_batch.int())
                f1.update(y_pred_prob.view(-1), y_batch.int())
                accuracy.update(y_pred_prob.view(-1), y_batch.int())

            avg_loss = epoch_loss / len(train_loader)
            
            # Print progress first, last, and ten times
            if self.verbose:
                if (e % int(self.max_epochs/10) == 0 or e == self.max_epochs) or e == 1:
                    print(f'Epoch {e+0:03}: |',
                          f'Loss: {avg_loss:.5f} |',
                          f'Accuracy: {accuracy.compute().item():.3f} |',
                          f'Recall: {recall.compute().item():.3f} |',
                          f'Precision: {precision.compute().item():.3f} |',
                          f'F1: {f1.compute().item():.3f}')
        
            # Stop criterion
            if abs(avg_loss - avg_loss_old) <= self.tol:
                count_iter += 1
                if self.verbose:
                    print(f'Current Epoch: {e} |',
                          f'Loss change: {abs(avg_loss - avg_loss_old):.1e} |',
                          f'Consecutive stop iterations: {count_iter}')
                if count_iter == self.n_consec_iter:
                    if self.verbose: print('Early stop criterion reached.')
                    break
            else:
                count_iter = 0
            avg_loss_old = avg_loss

    # Modified version of .predict_proba() for pd.DataFrame
    # def predict_proba(self, test_loader: DataLoader):
    def predict_proba(self, X_test: pd.DataFrame):
        device = next(self.parameters()).device  # Get the device from the model

        # Test DataLoader
        test_data = TestData(torch.tensor(X_test.values, dtype=torch.float32))
        test_loader = DataLoader(dataset=test_data, batch_size=self.batch_size, shuffle=False)
        # print(f'Test loader size: {len(test_loader)} batches')

        # Make predictions
        y_pred_prob_list = []
        self.eval()
        with torch.no_grad():
            for X_batch in test_loader:
                # Send data to device
                X_batch = X_batch.to(device)
        
                # Forward pass
                y_logit = self(X_batch)
                y_pred_prob_list.append(torch.sigmoid(y_logit))
        
        # Reshape for compatibility with sklearn
        p = torch.cat(y_pred_prob_list, dim=0).squeeze().detach().numpy()
        return np.concatenate([1-p.reshape(-1, 1), p.reshape(-1, 1)], axis=1)
        
    def predict(self, X_test: pd.DataFrame):
        y_pred_prob = self.predict_proba(X_test)[:, 1]
        return (y_pred_prob >= 0.5).astype(int)


