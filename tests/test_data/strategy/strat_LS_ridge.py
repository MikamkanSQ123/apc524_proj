from simple_backtester import Strategy
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

class Strat_Long_Short_Ridge(Strategy):
    def evaluate(self):
        # Set random seed for reproducibility
        seed = 999

        # Collect all matrices from strategy.features
        X_test_data_matrices = []
        X_train_data_matrices = []
        y_train = getattr(self.features, 'return')[-2,]
        # index -2 is to avoid lookahead bias, because the last row (-1) is the forward return to predict
        # Define moving average windows
        ma_windows = self.parameters.ma_windows
        # Loop through all attributes in strategy.features
        for feature_name in vars(self.features):
                
            matrix = getattr(self.features, feature_name)  # Get the matrix
            if feature_name == 'return':
                # use the lagged raw return as X
                X_train_data_matrices.append(matrix[-3,:].reshape(-1,1))
                X_test_data_matrices.append(matrix[-2,:].reshape(-1,1))
            else:
                matrix = (matrix - np.mean(matrix, axis=0)) / np.std(matrix, axis=0)

                # Prepare X matrix (moving averages)
                matrix_train = np.vstack(
                    [np.mean(matrix[-w-1:-1], axis=0) for w in ma_windows]
                ).T  # Transpose to make it n_samples x n_features
                matrix_test = np.vstack(
                    [np.mean(matrix[-w:], axis=0) for w in ma_windows]
                ).T  # Transpose to make it n_samples x n_features
                X_train_data_matrices.append(matrix_train)
                X_test_data_matrices.append(matrix_test)
                

        # Concatenate all matrices along axis=1
        X_train = np.concatenate(X_train_data_matrices, axis=1)
        X_test = np.concatenate(X_test_data_matrices, axis=1)

        # fillna with 0
        index_nan = np.isnan(X_train).any(axis=1) | np.isnan(y_train)
        X_train[index_nan] = 0
        y_train[index_nan] = 0
        X_test[index_nan] = 0

        ridge_model = RidgeCV(fit_intercept=False, scoring= 'neg_mean_squared_error',
                      alphas=self.parameters.alphas, 
                      cv=KFold(n_splits=self.parameters.cv, shuffle=True, random_state=seed))
        
        ridge_model.fit(X_train, y_train)
        signal = ridge_model.predict(X_test)
        signal = (signal - np.mean(signal)) / np.std(signal)
        signal[index_nan] = 0
        
        return signal