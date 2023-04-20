import numpy as np
from GPy.util.linalg import tdot, pdinv, dpotrs
from GPy.util import diag
from GPy.inference.latent_function_inference.posterior import PosteriorExact as Posterior
log_2_pi = np.log(2*np.pi)


class GPRegression:
    def __init__(self, X, Y, kernel, noise_var):
        """__init__.
        :param X: (n, dimX)
        :param Y: (n, 1), or for nIns instances of datasets y: (n, nIns)
        """
        self.X = X
        self.Y = Y
        self.kern = kernel
        self.noise_var = noise_var

        self.K = None
        self.Kinvs = None
        self.posterior = None

        self.update_model()

    def set_Y(self, Y):
        self.Y = Y
        self.update_model()

    def update_model(self):
        YYT_factor = self.Y
        if self.K is None:
            self.K = self.kern.K(self.X)
            Ky = self.K.copy()
            diag.add(Ky, self.noise_var)
            self.Kinvs = pdinv_warn(Ky)

        # Notes:
        # inv, chol, inv chol, logdet of prior cov mat K
        # data type=d, (guess) Positive definite triangular solve Ax=B. Pass in the chol factor of A
        #   ie output A\inv B
        # Note: why Y.shape[1]? Basically Y.shape[1] instances of data observed on the same X. Just sums
        #   the log_marginals from all instances

        Wi, LW, LWi, W_logdet = self.Kinvs
        alpha, _ = dpotrs(LW, YYT_factor, lower=1)
        # log_marginal =  0.5*(-self.Y.size * log_2_pi - self.Y.shape[1] * W_logdet - np.sum(alpha * YYT_factor))
        self.posterior = Posterior(woodbury_chol=LW, woodbury_vector=alpha, K=self.K)
        self.alpha_col = alpha.reshape((-1,1))
        self.LWi = LWi


class OneDValueAndGradientPredictionGPR(GPRegression):
    def __init__(self, X, Y, kernel, noise_var):
        assert kernel.input_dim == 1
        super().__init__(X, Y, kernel, noise_var)

    def predictValue(self, xx):
        """
        xx: location to predict value mean and variance
        """
        k_vx = self.kern.K(xx, self.X)
        k_vv = self.kern.K(xx)
        b = np.dot(self.LWi, k_vx.T)
        m = np.dot(k_vx, self.alpha_col)
        V = k_vv - np.dot(b.T, b)
        return m, V

    def predictGradient(self, xx):
        """
        xx: location to predict gradient mean and variance
        """
        k_gx = self.kern.K(xx, self.X, drv=(0,))
        k_gg = self.kern.K(xx, xx, drv=(0,), drv2=(0,))
        b = np.dot(self.LWi, k_gx.T)
        m = np.dot(k_gx, self.alpha_col)
        V = k_gg - np.dot(b.T, b)
        return m, V

    def predictValueAndGradient(self, xx):
        """
        xx: location to predict mean and co-variance of value and gradient
        """
        k_vx = self.kern.K(xx, self.X)
        k_gx = self.kern.K(xx, self.X, drv=(0,))[1]
        k_vv = self.kern.K(xx, xx)[0,0]
        k_gg = self.kern.K(xx, xx, drv=(0,), drv2=(0,))[1][0,0]
        k_gv = self.kern.K(xx, xx, drv=(0,))[1][0,0]

        k_ux = np.concatenate([k_vx, k_gx], axis=0)
        k_uu = np.stack([[k_vv, k_gv], [k_gv, k_gg]])
        b = np.dot(self.LWi, k_ux.T)
        m = np.dot(k_ux, self.alpha_col)
        V = k_uu - np.dot(b.T, b)
        return m, V
    
    def plot_posterior(self, grid, ax, showTrn=False, xlb=None, xub=None, ylb=None, yub=None):
        grid = grid.flatten()
        N = grid.size
        m = np.full((N,), np.nan)
        V = np.full((N,), np.nan)
        m_grad = np.full((N,), np.nan)
        V_grad = np.full((N,), np.nan)
        for i in range(N):
            mo, Vo = self.predictValueAndGradient(grid[i].reshape((1,1)))
            m[i], V[i] = mo[0,0], Vo[0,0]
            m_grad[i], V_grad[i] = mo[1,0], Vo[1,1]
        ax[0].plot(grid, m, 'k-')
        ax[0].plot(grid, m+np.sqrt(V)*2, color='gainsboro')
        ax[0].plot(grid, m-np.sqrt(V)*2, color='gainsboro')
        if showTrn:
            ax[0].plot(self.X, self.Y, '+', markersize=8)
        ax[1].plot(grid, m_grad, 'k-')
        ax[1].plot(grid, m_grad+np.sqrt(V_grad)*2, color='gainsboro')
        ax[1].plot(grid, m_grad-np.sqrt(V_grad)*2, color='gainsboro')
        if showTrn:
            ax[1].plot(self.X, np.zeros(self.X.shape), '+', markersize=8)

class StationaryKernel:
    def __init__(self, input_dim, variance, lengthscale):
        self.input_dim = input_dim
        self.variance = variance
        if isinstance(lengthscale, (int, float)) or len(lengthscale) == 1:
            lengthscale = np.tile(lengthscale, input_dim)
        self.lengthscale = lengthscale

    def K(self, X, X2=None, drv=None):
        """
        Kernel function applied on inputs X and X2.
        K(X, X2) = K_of_r((X-X2)**2)
        """
        r = self.r(X, X2)
        
        if drv is None:
            K = self.K_of_r(r)
            return K

        drv_order = len(drv)
        if drv_order == 0:
            K = self.K_of_r(r)
            return K, K

        if drv_order == 1:
            K, d1K = self.K_of_r(r, drv=1)
            r, d1r = self.r(X, X2, drv=drv, cc_r=r)
            dK = d1K * d1r
            return K, dK

        elif drv_order == 2:
            K, d1K, d2K = self.K_of_r(r, drv=2)
            if drv[0] == drv[1]:
                r, d1r = self.r(X, X2, drv=(drv[0],), cc_r=r)
                r, d2r = self.r(X, X2, drv=drv, cc_r=r)
                dK = d2K * d1r**2 + d1K * d2r
            else:
                r, d1r_d0 = self.r(X, X2, drv=(drv[0],), cc_r=r)
                r, d1r_d1 = self.r(X, X2, drv=(drv[1],), cc_r=r)
                r, d2r = self.r(X, X2, drv=drv, cc_r=r)
                dK = d2K * d1r_d0 * d1r_d1 + d1K * d2r
            return K, dK
        else:
            raise


    def _unscaled_dist(self, X, X2=None):
        """
        Compute the Euclidean distance between each row of X and X2, or between
        each pair of rows of X if X2 is None.
        """
        #X, = self._slice_X(X)
        if X2 is None:
            Xsq = np.sum(np.square(X),1)
            r2 = -2.*tdot(X) + (Xsq[:,None] + Xsq[None,:])
            # util.diag.view(r2)[:,]= 0. # force diagnoal to be zero: sometime numerically a little negative
            r2 = np.clip(r2, 0, np.inf)
            return np.sqrt(r2)
        else:
            #X2, = self._slice_X(X2)
            X1sq = np.sum(np.square(X),1)
            X2sq = np.sum(np.square(X2),1)
            r2 = -2.*np.dot(X, X2.T) + (X1sq[:,None] + X2sq[None,:])
            r2 = np.clip(r2, 0, np.inf)
            return np.sqrt(r2)

    def r(self, X, X2=None, drv=None, cc_r=None):
        """
        Compute the scaled distance, r.
            r = \sqrt( \sum_{q=1}^Q (x_q - x'q)^2/l_q^2 )
            drv: derivative specifier, list of coordinates to be differentiated over,
                can be repeated (coord0, coord1,...)
        """
        if cc_r is None:
            if X2 is None:
                r = self._unscaled_dist(X/self.lengthscale, None)
            else:
                r = self._unscaled_dist(X/self.lengthscale, X2/self.lengthscale)
        else:
            r = cc_r
        
        if drv is None:
            return r
        
        ir = 1. / r
        if np.any(np.isinf(ir)):
            ir[np.isinf(ir)] = 0
            print('Warning: Derivatives of kernel matrix at distance=0 points involved.')
            # set to 0 but this may be incorrect

        drv_order = len(drv)
        
        if drv_order == 1:
            d = drv[0]
            dr = .5 * ir * 2*(X[:, d:d+1] - X2[:, d:d+1].T) / self.lengthscale[d]**2
            return (r, dr)

        elif drv_order == 2:
            if drv[0] == drv[1]:
                d = drv[0]
                dr = -.25 * ir **3 \
                    * (2*(X[:,d:d+1] - X2[:,d:d+1].T) / self.lengthscale[d]**2) ** 2 \
                    + .5 * ir  * 2 / self.lengthscale[d]**2
            else:
                d1, d2 = drv[0], drv[1]
                dr = -.25 * ir **3 \
                    * 2*(X[:,d1:d1+1] - X2[:,d1:d1+1].T) / self.lengthscale[d1]**2 \
                    * 2*(X[:,d2:d2+1] - X2[:,d2:d2+1].T) / self.lengthscale[d2]**2
            return (r, dr)
        else:
            raise


class RBF(StationaryKernel):
    def __init__(self, input_dim, variance, lengthscale):
        super().__init__(input_dim, variance, lengthscale)
        self.name = 'RBF'

    def K(self, X, X2=None, drv=None, drv2=None):
        """
        drv:  derivative type acting on the 1st argument:
        drv2: derivative type acting on the 2nd argument:
        Thus, returns D_(x) D_(x2) K(x,x2)
        """
        if self.input_dim >= 2:
            return super().K(X, X2, drv)

        ##############################
        # 1d RBF K method
        ##############################
        # print('use 1d rbf K method')
        assert(self.input_dim == 1)
        assert(X.shape[1] == 1)
        if X2 is not None:
            assert(X2.shape[1] == 1)
        if X2 is None:
            X2 = X

        x_dif = X - X2.T
        r_sqr = np.square( x_dif / self.lengthscale)
        A = np.exp(-0.5 * r_sqr)
        K = self.variance * A

        if drv is None:
            if drv2 is None:
                return K
            else:
                raise NotImplementedError

        else:
            drv_order = len(drv)
            drv2_order = len(drv2) if drv2 is not None else 0
            if drv_order == 0:
                if drv2_order == 0:
                    return K, K
                else:
                    raise NotImplementedError
            if drv_order == 1:
                if drv2_order == 0:
                    B = x_dif / (self.lengthscale**2)
                    dK = -K * B
                    return K, dK
                elif drv2_order == 1:
                    B = x_dif / (self.lengthscale**2) 
                    dK = K / (self.lengthscale**2) - K * np.square(B)
                    return K, dK
                else:
                    raise NotImplementedError
            if drv_order == 2:
                if drv2_order == 0:
                    B = x_dif / (self.lengthscale**2) 
                    dK = -K / (self.lengthscale**2) + K * np.square(B)
                    return K, dK
                else:
                    raise NotImplementedError

    def K_of_r(self, r, drv=None):
        A = np.exp(-0.5 * r**2)
        K = self.variance * A
        if drv is None:
            return K

        d1K = -self.variance * A * r
        if drv == 1:
            return K, d1K

        d2K = self.variance * A * (r**2 - 1)
        if drv == 2:
            return K, d1K, d2K


class Matern32(StationaryKernel):
    def __init__(self, input_dim, variance, lengthscale):
        super().__init__(input_dim, variance, lengthscale)
        self.name = 'Matern32'

    def K(self, X, X2=None, drv=None):
        if self.input_dim >= 2:
            return super().K(X, X2, drv)

        # print('use 1d matern32 K method')
        assert(self.input_dim == 1)
        assert(X.shape[1] == 1)
        if X2 is not None:
            assert(X2.shape[1] == 1)
        if X2 is None:
            X2 = X

        SQRT3 = np.sqrt(3)
        S = np.sign(X - X2.T)
        D = np.abs(X - X2.T)
        A = SQRT3 * D / self.lengthscale
        E = np.exp(-A)
        K = self.variance * (1. + A) * E

        if drv is None:
            return K

        drv_order = len(drv)
        if drv_order==0:
            return K,K
        if drv_order==1:
            dK = self.variance * SQRT3/self.lengthscale * S * E * (-A)
            return K,dK
        if drv_order==2:
            dK = self.variance * 3/(self.lengthscale**2) *E * (A-1)
            return K,dK


    def K_of_r(self, r, drv=None):
        A = np.exp(-np.sqrt(3) * r)
        K = self.variance * (1 + np.sqrt(3) * r) * A
        if drv is None:
            return K

        d1K = -self.variance * 3. * A * r
        if drv == 1:
            return K, d1K

        d2K = self.variance * 3. * A * (np.sqrt(3) * r - 1)
        if drv == 2:
            return K, d1K, d2K


class Matern52(StationaryKernel):
    def __init__(self, input_dim, variance, lengthscale):
        super().__init__(input_dim, variance, lengthscale)
        self.name = 'Matern52'

    def K(self, X, X2=None, drv=None):
        if self.input_dim >= 2:
            return super().K(X, X2, drv)

        # print('use 1d matern52 K method')
        assert(self.input_dim == 1)
        assert(X.shape[1] == 1)
        if X2 is not None:
            assert(X2.shape[1] == 1)
        if X2 is None:
            X2 = X

        SQRT5 = np.sqrt(5)
        X_DIF = X - X2.T
        S = np.sign(X_DIF)
        D = np.abs(X_DIF)
        A = SQRT5 * D / self.lengthscale
        E = np.exp(-A)
        Q = np.square(D / self.lengthscale)
        K = self.variance * (1. + A + 5./3. * Q) * E

        if drv is None:
            return K

        drv_order=len(drv)
        if drv_order == 0:
            return K,K
        if drv_order==1:
            dK = self.variance * E * (10./3. * X_DIF/(self.lengthscale**2) - SQRT5/self.lengthscale * S * (A + 5./3. * Q))
            return K,dK
        if drv_order==2:
            dK = self.variance * E/(self.lengthscale**2) * (-5./3. - 5./3.*A + 25./3.*Q)
            return K,dK


    def K_of_r(self, r, drv=None):
        A = np.exp(-np.sqrt(5) * r)
        K = self.variance * (1 + np.sqrt(5) * r + 5./3. * r**2) * A
        if drv is None:
            return K

        d1K = -self.variance * 5./3. * A * r * (1 + np.sqrt(5) * r)
        if drv == 1:
            return K, d1K

        d2K = -self.variance * 5./3. * A * (1 + np.sqrt(5) * r - 5. * r**2)
        if drv == 2:
            return K, d1K, d2K


def pdinv_warn(A):
    """pdinv_warn.
    (Guess) pos. def. matrix inverse
    pdinv tries jittering if psd fails numerically
    Here we make sure it is reported
    """
    try:
        Kinvs = pdinv(A, 0)
    except Exception as e:
        print(f'pdinv_warn exception: {e}')
        Kinvs = pdinv(A)

    return Kinvs


