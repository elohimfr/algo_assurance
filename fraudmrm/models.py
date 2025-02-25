import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryRecall, BinaryPrecision, BinaryF1Score, BinaryAccuracy


class BinaryMLP(nn.Module):
    def __init__(self, 
                 input_size: int,
                 layer1_size: int=None,
                 layer2_size: int=None,
                 layer3_size: int=None,
                 learning_rate: float=1e-3,
                 max_epochs: int=200,
                 tol: float=1e-4,
                 n_consec_iter: int=10,
                 verbose: bool=True,
                ) -> None:
        super().__init__()
        # Training parameters
        self.input_size = input_size
        self.layer1_size = layer1_size
        if self.layer1_size == None:
            self.layer1_size = int(2**np.ceil(np.log2(self.input_size)))
        self.layer2_size = layer2_size
        if self.layer2_size == None:
            self.layer2_size = int(self.layer1_size/2)
        self.layer3_size = layer3_size
        if self.layer3_size == None:
            self.layer3_size = max(16, int(self.layer2_size/2))
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tol = tol
        self.n_consec_iter = n_consec_iter
        self.verbose = verbose

        # Layers
        self.layer1 = nn.Linear(self.input_size, self.layer1_size)
        self.layer2 = nn.Linear(self.layer1_size, self.layer2_size)
        self.layer3 = nn.Linear(self.layer2_size, self.layer3_size)
        self.layer_out = nn.Linear(self.layer3_size, 1)
        # Activation, Dropout, and Batch Normalisation
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(self.layer1_size)
        self.batchnorm2 = nn.BatchNorm1d(self.layer2_size)
        self.batchnorm3 = nn.BatchNorm1d(self.layer3_size)

    def forward(self, x):
        # Layer 1
        x = self.relu(self.layer1(x))
        x = self.batchnorm1(x)
        x = self.dropout(x)
        # Layer 2
        x = self.relu(self.layer2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        # Layer 3
        x = self.relu(self.layer3(x))
        x = self.batchnorm3(x)
        x = self.dropout(x)
        # Output layer
        x = self.layer_out(x)

        return x

    def fit(self, train_loader: DataLoader):
        # Send model to device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
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
                loss = loss_function(y_logit, y_batch.view(-1, 1))
                epoch_loss += loss.item()
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                # Compute metrics
                recall.update(y_pred_prob.squeeze(), y_batch.int())
                precision.update(y_pred_prob.squeeze(), y_batch.int())
                f1.update(y_pred_prob.squeeze(), y_batch.int())
                accuracy.update(y_pred_prob.squeeze(), y_batch.int())

            avg_loss = epoch_loss / len(train_loader)
            
            # Print progress first, last, and ten times
            if self.verbose:
                if e % int(self.max_epochs/10) == 0 or e == 1 or e == self.max_epochs:
                    print(f'Epoch {e+0:03}: |',
                          f'Loss: {avg_loss:.5f} |',
                          f'Accuracy: {accuracy.compute().item():.3f} |',
                          f'Recall: {recall.compute().item():.3f} |',
                          f'Precision: {precision.compute().item():.3f} |',
                          f'F1: {f1.compute().item():.3f}')
        
            # Stop criterion
            if abs(avg_loss - avg_loss_old) <= self.tol:
                count_iter += 1
                print(f'Current Epoch: {e} |',
                      f'Loss change: {abs(avg_loss - avg_loss_old):.1e} |',
                      f'Consecutive stop iterations: {count_iter}')
                if count_iter == self.n_consec_iter:
                    print('Early stop criterion reached.')
                    break
            else:
                count_iter = 0
            avg_loss_old = avg_loss
        print(f'Epoch {e+0:03}: |',
              f'Loss: {avg_loss:.5f} |',
              f'Accuracy: {accuracy.compute().item():.3f} |',
              f'Recall: {recall.compute().item():.3f} |',
              f'Precision: {precision.compute().item():.3f} |',
              f'F1: {f1.compute().item():.3f}')
        
    def predict_proba(self, test_loader: DataLoader):
        device = next(self.parameters()).device  # Get the device from the model
        y_pred_prob_list = []
        self.eval()
        with torch.no_grad():
            for X_batch in test_loader:
                # Send data to device
                X_batch = X_batch.to(device)
        
                # Forward pass
                y_logit = self(X_batch)
                y_pred_prob_list.append(torch.sigmoid(y_logit))
        
        return torch.cat(y_pred_prob_list, dim=0).squeeze().detach().numpy()
        
    def predict(self, test_loader: DataLoader):
        y_pred_prob = self.predict_proba(test_loader)
        return (y_pred_prob >= 0.5).astype(int)





