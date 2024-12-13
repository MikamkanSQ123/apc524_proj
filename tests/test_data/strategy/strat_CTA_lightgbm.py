from simple_backtester import Strategy
import numpy as np
import lightgbm as lgb

class Strat_CTA_LightGBM(Strategy):
    def evaluate(self):
        # Set random seed for reproducibility
        seed = 42
        np.random.seed(seed)
        
        # Configure LightGBM parameters
        lgb_params = vars(self.parameters)
        lgb_params.update({
            'seed': seed,
            'bagging_seed': seed,
            'feature_fraction_seed': seed,
            'drop_seed': seed,
            'deterministic': True,
            'num_threads': 1
        })

        # Collect all matrices from strategy.features
        X_test_data_matrices = []
        X_train_data_matrices = []
        y_train = getattr(self.features, 'return')[:-2,].squeeze()

        # Loop through all attributes in strategy.features
        for feature_name in vars(self.features):
            matrix = getattr(self.features, feature_name)  # Get the matrix
            if feature_name == 'return':
                # use the lagged raw return as X
                X_train_data_matrices.append(matrix[:-2,:].reshape(-1,1))
                X_test_data_matrices.append(matrix[-2,:].reshape(-1,1))
            else:
                X_train_data_matrices.append(matrix[1:-1,:].reshape(-1,1))
                X_test_data_matrices.append(matrix[-1,:].reshape(-1,1))

        # Concatenate all matrices along axis=1
        X_train = np.concatenate(X_train_data_matrices, axis=1)
        X_test = np.concatenate(X_test_data_matrices, axis=1)

        # Prepare LightGBM dataset
        train_data = lgb.Dataset(X_train, label=y_train)

        # Train LightGBM model
        gbm = lgb.train(lgb_params, train_data, num_boost_round=100)

        # Predict
        signal = gbm.predict(X_test)

        return signal
