import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional
from scipy.spatial.distance import cdist

############################################################################
# Region: PredictingMachine
############################################################################
class PredictingMachine(ABC):
    def __init__(self, d: int, M: int):
        self.d = d # input dimensionality
        self.M = M # output dimensionality

        @abstractmethod
        def reset_model(self):
            """Resets the internal model state"""
            pass

        @abstractmethod
        def predict(self, x: np.ndarray) -> np.ndarray:
            """Predicts an output based on the given input vector/matrix"""
            pass

        @abstractmethod
        def confidence(self, x: np.ndarray) -> float:
            """Evaluates the confidence score of a prediction"""
            pass

        @abstractmethod
        def confidence_batch(self, X: np.ndarray) -> List[float]:
            """Evaluates the confidence scores for a batch of predictions"""
            pass

############################################################################
# Region: BatchLearningMachine
############################################################################
class BatchLearningMachine(PredictingMachine):
    def __init__(self, d: int, M: int):
        super().__init__(d, M)   # call within a child class to initialize the parent class

    @abstractmethod
    def build_model(self, X: np.ndarray, Y: np.ndarray):
        """Builds the model using a batch of data, where 
        X: (samples x features (d))
        Y: (samples x targets (M))"""
        pass

############################################################################
# Region: IncrementalLearningMachine
############################################################################
class IncrementalLearningMachine(PredictingMachine):
    def __init__(self, d: int, M: int):
        super().__init__(d, M)
    
    @abstractmethod
    def update_model(self, x: np.ndarray, y: np.ndarray):
        """Updates the model with a new input-output pair"""
        pass

    @abstractmethod
    def update_batch(self, X: np.ndarray, Y: np.ndarray):
        """Updates the model with a batch of input-output pairs"""
        pass

    @abstractmethod
    def downdate_model(self, x: np.ndarray, y: np.ndarray):
        """Removes the influence of a specific input-output pair"""
        pass

    @abstractmethod
    def get_model(self) -> List[object]:
        """Retrieves the internal state of the model. Extracts the matrices into a portable list, which can be saved into a file. 
        It is crucial to save A^-1 and B for incremental learning"""
        pass

    @abstractmethod
    def set_model(self, model_objects: List[object]):
        """Sets the internal state of the model. Takes the saved list and restores it to the trained state without the need to see the original data again."""
        pass

############################################################################
# Region: RidgeRegression (RR)
############################################################################
class RR(BatchLearningMachine):
    def __init__(self, d: int, M: int, lambda_val: float = 1.0, hyper_params: Optional[List[str]] = None):
        super().__init__(d, M)  # we're calling the parent BatchLearningMachine to set input and output dimensions

        if hyper_params: # we're checking if the hyper_params list was passed
            try:
                self.lambda_val = float(hyper_params[0])
            except (IndexError, ValueError):
                print("Warning: Invalid HyperParams provided. Defaulting to 1.0")
                self.lambda_val = 1.0
        else:
            self.lambda_val = lambda_val
    
        self._initialise()

    def _initialise(self):
        """Creates an Identity matrix of size d (dxd)"""
        self.I_d = np.eye(self.d)
        self.reset_model()
    
    def reset_model(self):  # to ensure the model's memory is fresh and it's ready for a new training session
        """Clears weights (W) and inverse matrix (A_inv)"""
        self.W = np.zeros((self.d, self.M))
        self.A_inv = np.zeros((self.d, self.d))

    def build_model(self, X: np.ndarray, Y: np.ndarray):
        """Computes the analytical solution for RR"""

        # Input validation (optional)
        if X.shape[1] != self.d:
            raise ValueError(f"Input dimension mismatch. Expected {self.d}, got {X.shape[1]}")
        
        XT = X.T
        regularization_term = self.lambda_val * self.I_d

        # A = X^T * X + lambda * I
        A = XT @ X + regularization_term

        # Calculate inverse of A
        self.A_inv = np.linalg.inv(A)

        # Calculate weights: W = Ainv * X^T * Y
        self.W = self.A_inv @ (XT @ Y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts output and handles both vectors and matrices"""
        return X @ self.W
    
    def confidence(self, x: np.ndarray) -> float:
        if x.ndim == 1:
            x = x.reshape(1, -1) # make it a row vector
        confidence_value = x @ self.A_inv @ x.T
        return float(confidence_value.item())
    
    def confidence_batch(self, X: np.ndarray) -> List[float]:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        conf_array = np.sum((X @ self.A_inv) * X, axis=1)
        return conf_array.tolist()
    
############################################################################
# Region: RidgeRegression with RandomFourierFeatures (RFFRR)
############################################################################   
class RFF_RR(BatchLearningMachine):
    def __init__(self, d: int, M: int, lambda_val: float = 1.0,
                 sigma: float = 1.0, D: int = 300,
                 hyper_params: Optional[List[str]] = None):
        """
        Initializes  the RFF_RR model

        Args:
            d: input dim
            M: output dim
            lambda_val: regularization param
            sigma: std for the RBF kernel
            D: no. of RFF
            hyper_params: list of strings from YAML or other file [lambda, sigma, D]
        """
        super().__init__(d, M)

        if hyper_params:
            try:
                self.lambda_val = float(hyper_params[0])
                self.sigma = float(hyper_params[1])
                self.D = int(hyper_params[2])
            except (IndexError, ValueError):
                print("Warning: Invalid HyperParams. Using defaults.")
                self.lambda_val = lambda_val
                self.sigma = sigma
                self.D = D
        else:
            self.lambda_val = lambda_val
            self.sigma = sigma
            self.D = D
        
        self._initialise()
    
    def _initialise(self):
        """Sets up random matrices for RFF (Omega and Beta) and the internal linear RR"""
        self.rr = RR(self.D, self.M, self.lambda_val)

        # Take Omega from the normal distribution with (mean, std) = (0, sigma)
        self.omega = np.random.normal(loc=0.0, scale=self.sigma, size=(self.D, self.d))

        # Take Beta from the uniform distribution (-pi, pi)
        self.beta = np.random.uniform(-np.pi, np.pi, size=(self.D,))

    def _phi(self, X: np.ndarray) -> np.ndarray:
        """Maps input space into RFF space. Handles both vectors and matrices"""
        if X.ndim == 1:
            X = X.reshape(1, -1) # row vector

        projection = X @ self.omega.T

        # Broadcasting automatically adds vector (D,) to every row of (N, D)
        projection += self.beta

        return np.sqrt(2.0 / self.D) * np.cos(projection)  # added np.sqrt(2.0 / self.D)
    
    def build_model(self, X: np.ndarray, Y: np.ndarray):
        """Maps X into RFF space and then train the RR model"""
        X_rff = self._phi(X)

        self.rr.build_model(X_rff, Y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_rff = self._phi(X)

        return self.rr.predict(X_rff)
    
    def confidence(self, x: np.ndarray) -> float:
        x_rff = self._phi(x)
        return self.rr.confidence(x_rff)
    
    def confidence_batch(self, X: np.ndarray) -> List[float]:
        X_rff = self._phi(X)
        return self.rr.confidence_batch(X_rff)
    
    def reset_model(self):
        """Resets the internal RR model"""
        self.rr.reset_model()

############################################################################
# Region: IncrementalRidgeRegression (iRR)
############################################################################
class iRR(IncrementalLearningMachine):
    def __init__(self, d: int, M: int, lambda_val: float = 1.0,
                forgetting_factor: float = 1.0,  
                hyper_params: Optional[List[str]] = None):
        super().__init__(d, M)
    
        if hyper_params:
            try:
                self.lambda_val = float(hyper_params[0])
                if len(hyper_params) > 1:
                    self.forgetting_factor= float(hyper_params[1])
                else:
                    self.forgetting_factor= forgetting_factor
            except (IndexError, ValueError):
                print("Warning: Invalid HyperParams. Defaulting lambda to 1.0")
                self.lambda_val = 1.0
                self.forgetting_factor= 1.0
        else:
            self.lambda_val = lambda_val
            self.forgetting_factor = forgetting_factor   
        
        self._initialise()
    
    def _initialise(self):
        self.reset_model()

    def reset_model(self):
        """Resets A_inv to (1/lambda)*I and B to 0"""
        initial_A_inv = (1.0 / self.lambda_val) * np.eye(self.d)
        initial_B = np.zeros((self.d, self.M))

        self.set_model([initial_A_inv, initial_B])
    
    def update_model(self, x: np.ndarray, y: np.ndarray):
        """Sherman-Morrison update"""

        x_col = x.reshape(-1, 1) # Shape (d, 1)
        y_col = y.reshape(-1, 1) # Shape (M, 1)

        # Sherman-Morrison formula
        denominator = self.forgetting_factor + (x_col.T @ self.A_inv @ x_col).item()
        numerator = self.A_inv @ x_col @ x_col.T @ self.A_inv

        # Update A_inv
        self.A_inv = (1 / self.forgetting_factor) * (self.A_inv - (numerator / denominator))

        # Update B
        self.B = self.forgetting_factor * self.B + (x_col @ y_col.T)

        # Update W
        self.W = self.A_inv @ self.B
    
    def update_batch(self, X: np.ndarray, Y: np.ndarray):
        """Block update using exact matrix inversion for the batch, with corrected exponential decay."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)
            
        N = X.shape[0] # get the batch size
        
        # Calculate the decay factor for the entire batch
        batch_forgetting_factor = self.forgetting_factor ** N
        
        A_old = np.linalg.inv(self.A_inv)
        
        # Apply exponential weighting to the samples within the batch itself
        if self.forgetting_factor < 1.0:
            powers = np.arange(N - 1, -1, -1, dtype=float)
            w_diag = self.forgetting_factor ** powers
            
            # X.T @ W @ X
            X_weighted = X * w_diag[:, np.newaxis]
            A_new = batch_forgetting_factor * A_old + X_weighted.T @ X
            self.B = batch_forgetting_factor * self.B + X_weighted.T @ Y
        else:
            # If forgetting_factor == 1.0, skip the weighting overhead
            A_new = A_old + X.T @ X
            self.B = self.B + X.T @ Y
        
        self.A_inv = np.linalg.inv(A_new)
        self.W = self.A_inv @ self.B
    
    def downdate_model(self, x: np.ndarray, y: np.ndarray):
        raise NotImplementedError("Incremental Ridge Regression: downdating not yet implemented.")
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            # if x is a vector (d,)
            return x @ self.W
        # if x is a batch (N, d)
        return x @ self.W

    def confidence(self, x: np.ndarray) -> float:
        x = x.reshape(1, -1)
        conf = x @ self.A_inv @ x.T
        return float(conf.item())
    
    def confidence_batch(self, X: np.ndarray) -> List[float]:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        conf_array = np.sum((X @ self.A_inv) * X, axis=1)
        return conf_array.tolist()
    
    def get_model(self) -> List[object]:
        return [self.A_inv, self.B]
    
    def set_model(self, model_objects: List[object]):
        self.A_inv = model_objects[0]
        self.B = model_objects[1]
        # Sync W when state changes
        self.W = self.A_inv @ self.B

    # Extension for first training the model on some initial batch data and only then learn incrementaly
    def start_batch(self, X: np.ndarray, Y: np.ndarray):
        batch_learner = RR(d=self.d, M=self.M, lambda_val=self.lambda_val)
        batch_learner.build_model(X, Y)
        
        A_inv_initial = batch_learner.A_inv

        # B is not stored, so we compute it
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        
        B_initial = X.T @ Y
        self.set_model([A_inv_initial, B_initial])
        print(f"Initial training on batch data is completed.")

############################################################################
# Region: IncrementalRidgeRegression with RandomFourierFeatures (iRFFRR)
############################################################################
class iRFF_RR(IncrementalLearningMachine):
    def __init__(self, d: int, M: int, lambda_val: float = 1.0,
                 sigma: float = 1.0, D: int = 300,
                 forgetting_factor: float = 1.0,
                 hyper_params: Optional[List[str]] = None):
    
        super().__init__(d, M)
    
        if hyper_params:
            try:
                # Expects [lambda, sigma, D]
                self.lambda_val = float(hyper_params[0])
                self.sigma = float(hyper_params[1])
                self.D = int(hyper_params[2])
                self.forgetting_factor = float(hyper_params[3]) if len(hyper_params) > 3 else forgetting_factor
            except (IndexError, ValueError):
                print("Warning: Invalid HyperParams. Using defaults.")
                self.lambda_val = lambda_val
                self.sigma = sigma
                self.D = D
                self.forgetting_factor = forgetting_factor
        else:
            self.lambda_val = lambda_val
            self.sigma = sigma
            self.D = D
            self.forgetting_factor = forgetting_factor

        self._initialise()

    def _initialise(self):
        self.irr = iRR(d=self.D, M=self.M, lambda_val=self.lambda_val, forgetting_factor=self.forgetting_factor)
        self.omega = np.random.normal(loc=0.0, scale=self.sigma, size=(self.D, self.d))
        self.beta = np.random.uniform(-np.pi, np.pi, size=(self.D,))

    def _phi(self, x: np.ndarray) -> np.ndarray:
        if x.ndim ==1:
            x_in = x.reshape(1, -1)
        else:
            x_in = x
        
        projection = x_in @ self.omega.T + self.beta
        return np.sqrt(2.0 / self.D) * np.cos(projection) # added np.sqrt(2.0 / self.D) for normalization
    
    def update_model(self, x: np.ndarray, y: np.ndarray):
        features = self._phi(x).flatten()
        self.irr.update_model(features, y)
    
    def update_batch(self, X: np.ndarray, Y: np.ndarray):
        features = self._phi(X)
        self.irr.update_batch(features, Y)
    
    def downdate_model(self, x: np.ndarray, y: np.ndarray):
        raise NotImplementedError("Incremental RFF: downdating not yet implemented.")
    
    def reset_model(self):
        self.irr.reset_model()

    def predict(self, x: np.ndarray) -> np.ndarray:
        features = self._phi(x)
        return self.irr.predict(features)
    
    def confidence(self, x: np.ndarray) -> float:
        features = self._phi(x).flatten()
        return self.irr.confidence(features)
    
    def confidence_batch(self, X: np.ndarray) -> List[float]:
        features = self._phi(X)
        return self.irr.confidence_batch(features)
    
    def get_model(self) -> List[object]:
        """Returns full state: [IRR_A_inv, IRR_B, Omega, Beta]"""
        return self.irr.get_model() + [self.omega, self.beta]
    
    def set_model(self, model_objects: List[object]):
        """Restores state"""
        irr_state = model_objects[:2]
        self.irr.set_model(irr_state)

        self.omega = model_objects[2]
        self.beta = model_objects[3]

    # Extension for first training the model on some initial batch data and only then learn incrementaly
    def start_batch(self, X: np.ndarray, Y: np.ndarray):
        # This uses the existing Omega/Beta so features are consistent
        Phi = self._phi(X)

        # Standard batch RR
        batch_learner = RR(d=self.D, M=self.M, lambda_val=self.lambda_val)
        batch_learner.build_model(Phi, Y)
        A_inv_initial = batch_learner.A_inv
        B_initial = Phi.T @ Y

        self.irr.set_model([A_inv_initial, B_initial])
        print(f"Initial training on batch data is completed.")

############################################################################
# Region: Exact Kernel Ridge Regression (ExactKRR)
############################################################################
class ExactKRR(IncrementalLearningMachine):
    """
    Exact Kernel Ridge Regression using the RBF kernel.
    Serves as the 'best-case' exact baseline compared to iRFF_RR.
    Retrains on all accumulated history at each update.
    """
    def __init__(self, d: int, M: int, lambda_val: float = 1.0,
                 sigma: float = 1.0, forgetting_factor: float = 1.0, hyper_params: Optional[List[str]] = None):
        super().__init__(d, M)
        
        if hyper_params:
            try:
                self.lambda_val = float(hyper_params[0])
                self.sigma = float(hyper_params[1])
                self.forgetting_factor = float(hyper_params[2]) if len(hyper_params) > 2 else forgetting_factor
            except (IndexError, ValueError):
                print("Warning: Invalid HyperParams. Using defaults.")
                self.lambda_val = lambda_val
                self.sigma = sigma
                self.forgetting_factor = forgetting_factor
        else:
            self.lambda_val = lambda_val
            self.sigma = sigma
            self.forgetting_factor = forgetting_factor
            
        self.gamma = (self.sigma ** 2) / 2.0
        self._initialise()
        
    def _initialise(self):
        self.reset_model()
        
    def reset_model(self):
        self.X_history = np.empty((0, self.d))
        self.Y_history = np.empty((0, self.M))
        self.alpha = None
        self.K_reg_inv = None
        self.warm_batch_size = 0
        
    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        dist_sq = cdist(X1, X2, metric='sqeuclidean')
        return np.exp(-self.gamma * dist_sq)
        
    def _retrain(self):
        N = self.X_history.shape[0]
        if N == 0:
            return
            
        # Prune forgotten history to prevent numerical overflow and keep retrains extremely fast
        if self.forgetting_factor < 1.0:
            max_history = int(np.ceil(np.log(1e-5) / np.log(self.forgetting_factor))) # calculating the max history size based on the forgetting factor and a small threshold (1e-5)
            if N > max_history:
                drop_count = N - max_history
                self.X_history = self.X_history[drop_count:]
                self.Y_history = self.Y_history[drop_count:]
                self.warm_batch_size = max(0, self.warm_batch_size - drop_count)
                N = self.X_history.shape[0]
                
        K = self._rbf_kernel(self.X_history, self.X_history)
        
        # Apply exponential weighting for the forgetting factor (W^-1)
        if self.forgetting_factor < 1.0:
            inc_count = N - self.warm_batch_size
            powers = np.empty(N, dtype=float)
            if self.warm_batch_size > 0:   # if there is a warm batch, we set the first warm_batch_size elements to inc_count, so that they have the same power for the forgetting factor
                powers[:self.warm_batch_size] = inc_count
            if inc_count > 0:  # if there are incremental samples, we set the remaining elements to decreasing powers
                powers[self.warm_batch_size:] = np.arange(inc_count - 1, -1, -1)
                
            w_inv = (1.0 / self.forgetting_factor) ** powers
            decayed_lambda = self.lambda_val * (self.forgetting_factor ** inc_count)
            reg_matrix = np.diag(decayed_lambda * w_inv)
        else:
            reg_matrix = self.lambda_val * np.eye(N)
            
        self.K_reg_inv = np.linalg.inv(K + reg_matrix)
        self.alpha = self.K_reg_inv @ self.Y_history
        
    def update_model(self, x: np.ndarray, y: np.ndarray):
        x_in = x.reshape(1, -1)
        y_in = y.reshape(1, -1)
        self.X_history = np.vstack([self.X_history, x_in])
        self.Y_history = np.vstack([self.Y_history, y_in])
        self._retrain()
        
    def update_batch(self, X: np.ndarray, Y: np.ndarray):
        self.X_history = np.vstack([self.X_history, X])
        self.Y_history = np.vstack([self.Y_history, Y])
        self._retrain()
        
    def start_batch(self, X: np.ndarray, Y: np.ndarray):
        self.warm_batch_size = X.shape[0]
        self.update_batch(X, Y)
        print("Initial training on batch data is completed (ExactKRR).")
        
    def downdate_model(self, x: np.ndarray, y: np.ndarray):
        raise NotImplementedError("ExactKRR: downdating not implemented.")
        
    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.alpha is None:
            return np.zeros((x.shape[0] if x.ndim > 1 else 1, self.M))
        if x.ndim == 1:
            x = x.reshape(1, -1)
        K_xX = self._rbf_kernel(x, self.X_history)
        return K_xX @ self.alpha
        
    def confidence(self, x: np.ndarray) -> float:
        N = self.X_history.shape[0]
        inc_count = N - self.warm_batch_size
        current_lambda = self.lambda_val * (self.forgetting_factor ** inc_count) if self.forgetting_factor < 1.0 else self.lambda_val
        
        if self.K_reg_inv is None:
            return 1.0 / current_lambda
        if x.ndim == 1:
            x = x.reshape(1, -1)
        K_xX = self._rbf_kernel(x, self.X_history)
        val = 1.0 - (K_xX @ self.K_reg_inv @ K_xX.T).item()
        return float(max(0.0, val) / current_lambda)
        
    def confidence_batch(self, X: np.ndarray) -> List[float]:
        N = self.X_history.shape[0]
        inc_count = N - self.warm_batch_size
        current_lambda = self.lambda_val * (self.forgetting_factor ** inc_count) if self.forgetting_factor < 1.0 else self.lambda_val
        
        if self.K_reg_inv is None:
            return [1.0 / current_lambda] * X.shape[0]
        if X.ndim == 1:
            X = X.reshape(1, -1)
        K_xX = self._rbf_kernel(X, self.X_history)
        vals = 1.0 - np.sum((K_xX @ self.K_reg_inv) * K_xX, axis=1)
        #vals = np.sum(1.0 - (K_xX @ self.K_reg_inv) * K_xX, axis=1)
        return (np.maximum(0.0, vals) / current_lambda).tolist()
        
    def get_model(self) -> List[object]:
        return [self.X_history, self.Y_history, self.alpha, self.K_reg_inv, self.warm_batch_size]
        
    def set_model(self, model_objects: List[object]):
        self.X_history, self.Y_history, self.alpha, self.K_reg_inv, self.warm_batch_size = model_objects


############################################################################
# Region: Preprocessing
############################################################################
class RollingStandardizer:
    """
    Rolling, online Channel-Wise Standardizer using Exponential Moving Average (EMA).
    Mitigates amplitude drift by continuously mapping input features to mean 0 and variance 1.
    """
    def __init__(self, d: int, alpha: float = 0.001, epsilon: float = 1e-8):
        self.d = d
        self.alpha = alpha
        self.epsilon = epsilon
        self.mean = np.zeros(d)
        self.var = np.ones(d)
        self.initialized = False

    def fit_batch(self, X: np.ndarray):
        """Warm-start the standardizer with batch data."""
        self.mean = np.mean(X, axis=0)
        self.var = np.var(X, axis=0)
        self.initialized = True

    def update(self, x: np.ndarray):
        """Update rolling statistics with a single sample."""
        if x.ndim == 2:
            x = x.flatten()
        if not self.initialized:
            self.mean = np.copy(x)
            self.var = np.ones(self.d)
            self.initialized = True
            return

        delta = x - self.mean
        self.mean += self.alpha * delta
        self.var = (1.0 - self.alpha) * (self.var + self.alpha * (delta ** 2))

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardize the input using current rolling stats."""
        return (X - self.mean) / (np.sqrt(self.var) + self.epsilon)

    def update_and_transform_batch(self, X: np.ndarray) -> np.ndarray:
        """Update stats and transform a batch sample-by-sample."""
        X_transformed = np.zeros_like(X, dtype=float)
        for i in range(X.shape[0]):
            self.update(X[i])
            X_transformed[i] = self.transform(X[i])
        return X_transformed


############################################################################
# Test Usage of RFFRR
############################################################################        
if __name__ == "__main__":
    # Non-linear data
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]])
    Y = np.array([[1.0], [4.0], [9.0], [16.0], [25.0], [36.0], [49.0]])

    D = 500
    model = RFF_RR(d=1, M=1, lambda_val=0.1, sigma=1.0, D=500)

    model.build_model(X, Y)

    test_input = np.array([2.5])
    prediction = model.predict(test_input)

    conf = model.confidence(test_input)

    print(f"Prediction for {test_input}: {prediction}")
    print(f"Confidence score: {conf}")
