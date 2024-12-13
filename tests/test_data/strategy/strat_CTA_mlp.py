from simple_backtester import Strategy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, dropout_rate):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class Strat_CTA_MLP(Strategy):
    def evaluate(self):
        # Set random seed for reproducibility
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Collect all matrices from strategy.features
        X_data_matrices = []
        y_data = getattr(self.features, 'return')[:-2]

        # Loop through all attributes in strategy.features
        for feature_name in vars(self.features):
            matrix = getattr(self.features, feature_name)
            if feature_name == 'return':
                X_data_matrices.append(matrix[:-2].reshape(-1, 1))
            else:
                X_data_matrices.append(matrix[1:-1].reshape(-1, 1))

        # Concatenate all matrices along axis=1
        X_data = np.concatenate(X_data_matrices, axis=1)

        # Standardize the data
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()
        X_data = X_scaler.fit_transform(X_data)
        y_data = y_scaler.fit_transform(y_data.reshape(-1, 1))

        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X_data, dtype=torch.float32)
        y_tensor = torch.tensor(y_data, dtype=torch.float32)

        # Prepare DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)

        def seed_worker(worker_id):
            np.random.seed(seed)
            torch.manual_seed(seed)

        dataloader = DataLoader(
            dataset,
            batch_size=self.parameters.batch_size,
            shuffle=True,
            num_workers=0,
            worker_init_fn=seed_worker
        )

        # Initialize model, loss function, and optimizer
        model = MLP(self.parameters.input_dim, self.parameters.hidden_layers, self.parameters.output_dim, self.parameters.dropout_rate)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.parameters.learning_rate)

        # Train the model
        model.train()
        for epoch in range(self.parameters.epochs):
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Add weight penalty
                l2_penalty = sum(param.pow(1.0).sum() for param in model.parameters())
                loss += self.parameters.penalty * l2_penalty
                
                loss.backward()
                optimizer.step()

        # Prepare test data
        X_test_data_matrices = []
        for feature_name in vars(self.features):
            matrix = getattr(self.features, feature_name)
            if feature_name == 'return':
                X_test_data_matrices.append(matrix[-2].reshape(-1, 1))
            else:
                X_test_data_matrices.append(matrix[-1].reshape(-1, 1))
        X_test = np.concatenate(X_test_data_matrices, axis=1)

        # Standardize the test data using the same parameters
        X_test = X_scaler.transform(X_test)

        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

        # Predict
        model.eval()
        with torch.no_grad():
            signal = model(X_test_tensor).numpy()

        # Inverse transform the signal
        signal = y_scaler.inverse_transform(signal).squeeze()
        return signal
