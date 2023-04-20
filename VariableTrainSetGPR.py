import numpy as np
from MyGPR import pdinv_warn
from GPy.util.linalg import dpotrs
from GPy.util import diag
from scipy.linalg import solve_toeplitz


class VariableTrainSetGPR:
    def predict(self, y):
        """predict
        compute test functional conditional mean
        :param y: (n, 1), or for nIns instances of datasets y: (n, nIns)
        """
        assert y.shape[0] == self.X.shape[0]
        return np.dot(self.w, y)
    

class ToeplitzGPR(VariableTrainSetGPR):
    def __init__(self, X, kernel, noise_var, KTsTr=None):
        """__init__.
        :param X: (n, dimX)
        """
        self.X = X
        self.kern = kernel
        self.noise_var = noise_var

        self.c_K = None  # column 1 of K_XX
        self.w = None  # conditional weight vector: E[L_test(f)|X,y] = np.dot(self.w,y)       
        self.KTsTr = KTsTr # cov[L_test(f), L_train(f)]. Or special case: cov[f(x_ts), f(x_tr)]
        if self.KTsTr is not None:
            self.update_model()
            
    def update_model(self):
        if self.c_K is None:
            self.c_K = self.kern.K(self.X, self.X[0:1,:])
            self.c_K[0,0] += self.noise_var
        assert self.KTsTr is not None
        self.w = solve_toeplitz(self.c_K, self.KTsTr.flatten()).reshape((1,-1))


class StandardGPR(VariableTrainSetGPR):
    def __init__(self, X, kernel, noise_var, KTsTr=None):
        """__init__.
        :param X: (n, dimX)
        """
        self.X = X
        self.kern = kernel
        self.noise_var = noise_var

        self.K = None
        self.w = None  # conditional weight vector: E[L_test(f)|X,y] = np.dot(self.w,y)       
        self.KTsTr = KTsTr # cov[L_test(f), L_train(f)]. Or special case: cov[f(x_ts), f(x_tr)]
        if self.KTsTr is not None:
            self.update_model()
        
    def update_model(self):
        if self.K is None:
            self.K = self.kern.K(self.X)
            Ky = self.K.copy()
            diag.add(Ky, self.noise_var)
            self.Kinvs = pdinv_warn(Ky)

        # inv, chol, inv chol, logdet of prior cov mat K
        Wi, LW, LWi, W_logdet = self.Kinvs
        assert self.KTsTr is not None
        self.w, _ = dpotrs(LW, self.KTsTr.reshape((-1,1)), lower=1)
        self.w = self.w.reshape((1,-1))
        
