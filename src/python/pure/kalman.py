import os
import sys
import numpy as np
import scipy.stats

# This file was created with the idea to be
# a mockup class hierarchy for a C++ implementation
# Therefore, it is not totally Pythonic.
# Sorry if you expected other thing
# The same tests will be translated in C++

#####################################################################
# Stats helpers
#####################################################################
def _inv(X):
    #try:
    #    return np.linalg.inv(X)
    #except:
    #    return np.linalg.pinv(X)
    return np.linalg.pinv(X)

def _mvn_probability(x, mean, cov):
    return np.exp(-0.5 * _dot((x - mean).T, _inv(cov), (x - mean))) / np.sqrt(2, np.pi * np.linalg.det(cov))

def _mvn_logprobability(x, mean, cov):
    return (-0.5 * _dot((x - mean).T, _inv(cov), (x - mean))) - 0.5 * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(cov))

def _mvn_sample(mean, cov):
    return scipy.stats.multivariate_normal.rvs(mean=mean.ravel(), cov=cov).reshape(mean.shape)

#####################################################################
# Math cross-platform functions
#####################################################################

def __create_noised_ones(L, M):
    return np.ones((L, M), dtype="f") + 0.05 * np.random.random((L, M))

def __create_noised_zeros(L, M):
    return np.zeros((L, M), dtype="f") + 0.05 * np.random.random((L, M))

def __create_noised_diag(L, M):
    return np.eye(L, M, dtype="f") + 0.05 * np.random.random((L, M))

def test_create_noised_ones():
    m = __create_noised_ones(1, 1)
    assert np.abs(m[0][0] - 1) < 0.1
    m = __create_noised_ones(10, 10)
    for i in range(10):
        for j in range(10):
            assert np.abs(m[i][j] - 1) < 0.1

def test_create_noised_zeros():
    m = __create_noised_zeros(1, 1)
    assert np.abs(m[0][0] - 0) < 0.1
    m = __create_noised_zeros(10, 10)
    for i in range(10):
        for j in range(10):
            assert np.abs(m[i][j] - 0) < 0.1

def test_create_noised_diag():
    m = __create_noised_diag(1, 1)
    assert np.abs(m[0][0] - 1) < 0.1
    m = __create_noised_diag(10, 10)
    for i in range(10):
        for j in range(10):
            if i == j:
                assert np.abs(m[i][j] - 1) < 0.1
            else:
                assert np.abs(m[i][j] - 0) < 0.1

def _sum_cube(X):
    return np.sum(X, axis=2)

def test_sum_cube():
    X = np.zeros((2, 2, 4))
    X[:, :, 0] = [[0, 1], [2, 3]]
    X[:, :, 1] = [[0, 0], [0, 1]]
    X[:, :, 2] = [[1, 1], [0, -1]]
    X[:, :, 3] = [[2, 0], [2, 0]]
    Y = _sum_cube(X)
    assert Y[0][0] == 3
    assert Y[0][1] == 2
    assert Y[1][0] == 4
    assert Y[1][1] == 3
    X = np.zeros((1, 1, 3))
    X[:, :, 0] = [1]
    X[:, :, 1] = [1]
    X[:, :, 2] = [2]
    Y = _sum_cube(X)
    assert Y[0][0] == 4

def _set_diag_values_positive(X):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i, j] = np.abs(X[i, j])

def _dot(*vars):
    p = vars[0]
    for i in range(1, len(vars)):
        p = p.dot(vars[i])
    return p

def test_dot():
    assert np.sum(_dot(
        np.array([[1, 0], [0, 1]]),
        np.array([[1], [10]])
    )) == 11
    assert np.sum(_dot(
        np.array([[1, 0], [0, 1]]),
        np.array([[1], [10]])
    ).shape) == 3
    assert np.sum(_dot(
        np.array([[1, 0], [0, 1]]),
        np.array([[1], [10]]),
        np.array([[5, 10]]),
    )) == (5 + 10 + 50 + 100)


def test_inv():
    X = np.array([[5, 10], [3, 6]])
    X1 = _dot(_dot(X, _inv(X)), X)
    X2 = _dot(X, _dot(_inv(X), X))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            assert np.abs(X1[i][j] - X[i][j]) < 0.1
    X = np.array([[5,10, 2], [3, 6, 10]])
    X1 = _dot(_dot(X, _inv(X)), X)
    X2 = _dot(X, _dot(_inv(X), X))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            assert np.abs(X1[i][j] - X[i][j]) < 0.1


def _t(X):
    return X.T

def _row(X, k):
    return X[k, :]

def _col(X, k):
    return X[:, k]

def _slice(X, k):
    return X[:, :, k]

def _set_row(X, k, v):
    X[k, :] = v

def _set_col(X, k, v):
    X[:, k] = v

def _set_slice(X, k, v):
    X[:, :, k] = v

def _nrows(X):
    return X.shape[0]

def _ncols(X):
    return X.shape[1]

def _nslices(X):
    return X.shape[2]

def _one_matrix(L, M):
    return np.ones((L, M), dtype="f")

def _zero_matrix(L, M):
    return np.zeros((L, M), dtype="f")

def _zero_cube(L, M, N):
    return np.zeros((L, M, N), dtype="f")

def _diag_matrix(L, M):
    return np.eye((L, M), dtype="f")

def _no_finite_to_zero(A):
    A[np.isinf(A)] = 0
    A[np.isinf(-A)] = 0
    A[np.isnan(A)] = 0

def _head_slices(X):
    return X[:, :, :X.shape[-1] - 1]

def _tail_slices(X):
    return X[:, :, X.shape[-1] - 1:]

#####################################################################
# Invariant SSM
#####################################################################

class SSMParameters:
    def __init__(self):
        self.F = None
        self.H = None
        self.Q = None
        self.R = None
        self.X0 = None
        self.P0 = None
        self.o = 0
        self.l = 0
    
    def set_dimensions(self, o, l):
        self.o = o
        self.l = l
    
    def latent_signal_dimension(self):
        return self.l

    def observable_signal_dimension(self):
        return self.o
    
    def show(self):
        print(self)
        print(" Transition matrix: ", self.F)
        print(" Observation matrix: ", self.H)
        print(" Latent var-covar matrix: ", self.Q)
        print(" Observation var-covar matrix: ", self.R)
        print(" Initial latent signal: ", self.X0)
        print(" Initial latent var-covar matrix: ", self.P0)

    def random_initialize(self, init_F=True, init_H=True, init_Q=True, init_R=True, init_X0=True, init_P0=True):
        if self.l is None:
            raise ValueError("Latent signal dimension is unset!")
        if self.o is None:
            raise ValueError("Observable signal dimension is unset!")
        if init_F: self.F = __create_noised_ones(self.l, self.l)
        if init_Q: self.Q = __create_noised_diag(self.l, self.l)
        if init_X0: self.X0 = __create_noised_ones(self.l, 1)
        if init_P0: self.P0 = __create_noised_diag(self.l, self.l)
        if init_H: self.H = __create_noised_ones(self.o, self.l)
        if init_R: self.R = __create_noised_diag(self.o, self.o)
    
    def estimate_error(self, Y):
        """
        const kmat& Y = observation_sequence;
        const kmat& H = observation_matrix; //C
        const kmat& X = hidden_state_estimations;
        const int T = _ncols(Y);
        kmat error = arma::zeros(_nrows(Y), _nrows(Y));
        for(int k = 0; k < T; k++){
            error += (Y.col(k) - H * X.col(k)) * (Y.col(k) - H * X.col(k)).t(); 
        }
        return error / T;
        """ 
        raise NotImplementedError("")
        # return squared error and covariance

    def simulate(self, N, X0=None, P0=None):
        if N == 0: return np.array([]), np.array([])
        if X0 is None:
            X0 = self.X0
        if P0 is None:
            P0 = self.P0
        R0 = _zero_matrix(_ncols(self.R), 1)
        Q0 = _zero_matrix(_ncols(self.Q), 1)
        X1 = _mvn_sample(X0, P0).reshape(-1, 1)
        Y1 = _dot(self.H, X1) + _mvn_sample(R0, self.R)
        X = [X1]
        Y = [Y1]
        for _ in range(1, N):
            X1 = _dot(self.F, X1) + _mvn_sample(Q0, self.Q)
            X.append(X1)
            Y1 = _dot(self.H, X1) + _mvn_sample(R0, self.R)
            Y.append(Y1)
        # observation_sequence
        Y = np.hstack(Y)
        # hidden_state_sequence
        X = np.hstack(X)
        return X, Y


def __create_params_ones_kx1(M, K=[100.0]):
    K = np.array([K]).T
    params = SSMParameters()
    params.F = np.array([[1-1e-10]])
    params.H = K
    params.Q = np.array([[0.01]])
    params.R = 0.01 * __create_noised_diag(_nrows(K), _nrows(K))
    params.X0 = np.array([M])
    params.P0 = np.array([[0.01]])
    params.set_dimensions(len(K), 1)
    return params

def test_simulations_params():
    params = __create_params_ones_kx1(-50, [10])
    _, y = params.simulate(100)
    assert np.abs(np.round(np.mean(_row(y, 0))) - -500)/100 < 0.1 * 500, "Failed simulation"

    params = __create_params_ones_kx1(2, [-100, 100])
    _, y = params.simulate(1)
    assert np.abs(np.round(np.mean(_row(y, 0))) - -200) < 0.1 * 200, "Failed simulation"
    assert np.abs(np.round(np.mean(_row(y, 1))) - 200) < 0.1 * 200, "Failed simulation"
    
    _, y = params.simulate(100)
    assert np.abs(np.round(np.mean(_row(y, 0))) - -200)/100 < 0.1 * 200, "Failed simulation"
    assert np.abs(np.round(np.mean(_row(y, 1))) - 200)/100 < 0.1 * 200, "Failed simulation"
    

class SSMEstimated:
    def __init__(self):
        self.X = None
        self.P = None
        self.ACV1 = None
    
    def signal(self): return self.X

    def variance(self): return self.P
    
    def autocovariance(self): return self.ACV1
    
    def init(self, dim, length, fill_ACV=False):
        self.X = _zero_matrix(dim, length)
        self.P = _zero_cube(dim, dim, length)
        if fill_ACV:
            self.ACV1 = _zero_cube(dim, dim, length - 1)



#####################################################################
# Kalman Filter
#####################################################################

"""
X[t] = F X[t-1] + N(0, Q)
Y[t] = H X[t] + N(0, R)
"""
class KalmanFilter:
    def __init__(self):
        self.parameters = SSMParameters()
        self.filtered_estimates = SSMEstimated()
        self.predicted_estimates = SSMEstimated()
        self._Y = None
    
    def T(self): return self.Y().shape[-1]
    
    def o(self): return self.parameters.o
    
    def obs_dim(self): return self.parameters.o
    
    def set_obs_dim(self, v): self.parameters.o = v
    
    def l(self): return self.parameters.l
    
    def latent_dim(self): return self.parameters.l
    
    def set_latent_dim(self, v): self.parameters.l = v

    def F(self): return self.parameters.F

    def set_F(self, v): self.parameters.F = v

    def H(self): return self.parameters.H
    
    def set_H(self, v): self.parameters.H = v

    def Q(self): return self.parameters.Q
    
    def set_Q(self, v): self.parameters.Q = v

    def R(self): return self.parameters.R
    
    def set_R(self, v): self.parameters.R = v

    def X0(self): return self.parameters.X0
    
    def set_X0(self, v): self.parameters.X0 = v

    def P0(self): return self.parameters.P0
    
    def set_P0(self, v): self.parameters.P0 = v

    def Y(self): return self._Y
    
    def set_Y(self, v): self._Y = v
    
    def Xf(self): return self.filtered_estimates.X

    def Pf(self): return self.filtered_estimates.P

    def Xp(self): return self.predicted_estimates.X

    def Pp(self): return self.predicted_estimates.P

    def loglikelihood(self):
        #https://pdfs.semanticscholar.org/6654/c13f556035c1ea9e7b6a7cf53d13c98af6e7.pdf
        log_likelihood = 0
        for k in range(1, self.T()):
            Sigma_k = _dot(self.H(), _slice(self.Pf(), k-1), _t(self.H())) + self.R()
            current_likelihood = _mvn_logprobability(_col(self.Y(), k), _dot(self.H(), _col(self.Xf(), k)), Sigma_k)
            if np.isfinite(current_likelihood):
                log_likelihood += current_likelihood
        return log_likelihood / (self.T() - 1)

    def verify_parameters(self):
        if self.l() == 0:
            raise ValueError("Observation sequence has no samples")

        if not(
            _nrows(self.Y()) == _nrows(self.H()) and
            _ncols(self.R()) == _nrows(self.Q()) and
            _ncols(self.R()) == _nrows(self.H())
        ):
            raise ValueError("There is no concordance in the dimension of observed signal.")

        if not(
            _nrows(self.P0()) == _ncols(self.P0()) and
            _nrows(self.X0()) == _ncols(self.P0()) and
            _nrows(self.X0()) == _ncols(self.H()) and
            _nrows(self.F()) == _ncols(self.H()) and
            _nrows(self.F()) == _ncols(self.Q()) and
            _ncols(self.F()) == _nrows(self.F()) and
            _ncols(self.Q()) == _nrows(self.Q())
        ):
            raise ValueError("There is no concordance in the dimension of latent signal.")

    def filter(self):
        self.verify_parameters()
        self.filtered_estimates.init(self.l(), self.T())
        self.predicted_estimates.init(self.l(), self.T())
        
        _set_col(self.Xp(), 0, self.X0())
        _set_slice(self.Pp(), 0, self.P0())

        k = 0
        # Kalman gain
        G = _dot(_slice(self.Pp(), k), _t(self.H()), _inv(_dot(self.H(), _slice(self.Pp(), k), _t(self.H())) + self.R()))
        # State estimate update
        _set_col(self.Xf(), k, _col(self.Xp(), k) + _dot(G, _col(self.Y(), k) - _dot(self.H(), _col(self.Xp(), k))))
        # Error covariance update
        _set_slice(self.Pf(), k, _slice(self.Pp(), k) - _dot(G, self.H(), _slice(self.Pp(), k)))

        for k in range(1, self.T()):
            # State estimate propagation
            _set_col(self.Xp(), k, _dot(self.F(), _col(self.Xf(), k - 1)))
            # Error covariance propagation
            _set_slice(self.Pp(), k, _dot(self.F(), _slice(self.Pf(), k-1), _t(self.F())) + self.Q())
            # Kalman gain
            G = _dot(_slice(self.Pp(), k), _t(self.H()), _inv(_dot(self.H(), _slice(self.Pp(), k), _t(self.H())) + self.R()))
            # State estimate update
            _set_col(self.Xf(), k, _col(self.Xp(), k) + _dot(G, _col(self.Y(), k) - _dot(self.H(), _col(self.Xp(), k))))
            # Error covariance update
            _set_slice(self.Pf(), k, _slice(self.Pp(), k) - _dot(G, self.H(), _slice(self.Pp(), k)))

def test_filter_1():
    params = __create_params_ones_kx1([-50], [10])
    x, y = params.simulate(100)
    kf = KalmanFilter()
    kf.set_Y(y)
    kf.parameters = params
    kf.filter()
    assert (np.abs(np.mean(kf.Xp()) - -50) <= 0.1 * 50), "Failed simulation: mean(X pred) != true mean"
    assert (np.abs(np.mean(kf.Xf()) - -50) <= 0.1 * 50), "Failed simulation: mean(X filter) != true mean"
    #assert np.round(np.std(kf.Xp()), 2) >= np.round(np.std(kf.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)"


def kalman_filter(Y, F, H, Q, R, X0, P0):
    kf = KalmanFilter()
    kf.set_F(F)
    kf.set_H(H)
    kf.set_Q(Q)
    kf.set_R(R)
    kf.set_X0(X0)
    kf.set_P0(P0)
    kf.set_Y(Y)
    kf.set_obs_dim(_nrows(Y))
    kf.set_latent_dim(_nrows(X0))
    kf.filter()
    return kf

def kalman_filter_from_parameters(Y, params):
    kf = KalmanFilter()
    kf.parameters = params
    kf.set_Y(Y)
    kf.set_obs_dim(_nrows(Y))
    kf.set_latent_dim(_nrows(params.X0))
    kf.filter()
    return kf

def test_filter_2():
    params = __create_params_ones_kx1([-50], [10])
    x, y = params.simulate(100)
    kf = kalman_filter(y, params.F, params.H, params.Q, params.R, params.X0, params.P0)
    assert (np.abs(np.mean(kf.Xp()) - -50) <= 0.1 * 50), "Failed simulation: mean(X pred) != true mean"
    assert (np.abs(np.mean(kf.Xf()) - -50) <= 0.1 * 50), "Failed simulation: mean(X filter) != true mean"
    #assert np.round(np.std(kf.Xp()), 2) >= np.round(np.std(kf.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)"
    kf = kalman_filter_from_parameters(y, params)
    assert (np.abs(np.mean(kf.Xp()) - -50) <= 0.1 * 50), "Failed simulation: mean(X pred) != true mean"
    assert (np.abs(np.mean(kf.Xf()) - -50) <= 0.1 * 50), "Failed simulation: mean(X filter) != true mean"
    #assert np.round(np.std(kf.Xp()), 2) >= np.round(np.std(kf.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)"


#####################################################################
# Kalman Smoother
#####################################################################

class KalmanSmoother(KalmanFilter):
    def __init__(self):
        super().__init__()
        self.smoothed_estimates = SSMEstimated()
    
    def Xs(self): return self.smoothed_estimates.X

    def Ps(self): return self.smoothed_estimates.P

    def Cs(self): return self.smoothed_estimates.ACV1

    def loglikelihood(self):
        log_likelihood = 0
        for k in range(1, self.T()):
            Sigma_k = _dot(self.H(), _slice(self.Ps(), k-1), _t(self.H())) + self.R()
            current_likelihood = _mvn_logprobability(_col(self.Y(), k), _dot(self.H(), _col(self.Xs(), k)), Sigma_k)
            if np.isfinite(current_likelihood):
                log_likelihood += current_likelihood
        return log_likelihood / (self.T() - 1)
    
    def smooth(self, filter=True):
        if filter:
            self.filter()
        
        self.smoothed_estimates.init(self.l(), self.T(), True)
        
        k = self.T() - 1
        _set_col(self.Xs(), k, _col(self.Xf(), k))
        _set_slice(self.Ps(), k, _slice(self.Pf(), k))
        k -= 1
        A_prev = None
        while k >= 0:
            A = _dot(_slice(self.Pf(), k), _t(self.F()), _inv(_slice(self.Pp(), k + 1)))
            _no_finite_to_zero(A)
            _set_slice(self.Ps(), k, _slice(self.Pf(), k) - _dot(A, _col(self.Xs(), k + 1) - _col(self.Xf(), k + 1), _t(A))) #Ghahramani
            _set_col(self.Xs(), k, _col(self.Xf(), k) + _dot(A, _col(self.Xs(), k + 1) - _col(self.Xf(), k + 1)))
            if k == self.T() - 2:
                G = _dot(_slice(self.Pp(), k + 1), _t(self.H()), _inv(_dot(self.H(), _slice(self.Pp(), k + 1), _t(self.H())) + self.R()))
                _set_slice(self.Cs(), k, _dot(self.F(), _slice(self.Pf(), k)) - _dot(G, self.H(), self.F(), _slice(self.Pf(), k)))
            else:
                _set_slice(self.Cs(), k, _dot(_slice(self.Pf(), k + 1), _t(A)) + _dot(A_prev, _t(_slice(self.Cs(), k + 1)) - _dot(self.F(), _slice(self.Pf(), k + 1)), _t(A_prev)))
            A_prev = A
            k -= 1

def test_smoother_1():
    params = __create_params_ones_kx1([-50], [10])
    x, y = params.simulate(100)
    kf = KalmanSmoother()
    kf.parameters = params
    kf.set_Y(y)
    kf.smooth()
    assert (np.abs(np.mean(kf.Xp()) - -50) <= 0.1 * 50), "Failed simulation: mean(X pred) != true mean"
    assert (np.abs(np.mean(kf.Xf()) - -50) <= 0.1 * 50), "Failed simulation: mean(X filter) != true mean"
    assert (np.abs(np.mean(kf.Xs()) - -50) <= 0.1 * 50), "Failed simulation: mean(X smooth) != true mean"
    #assert np.round(np.std(kf.Xp()), 2) >= np.round(np.std(kf.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)"
    #assert np.round(np.std(kf.Xf()), 2) >= np.round(np.std(kf.Xs()), 2), "Failed simulation: std(X filter) < std(X smooth)"
    

def test_smoother_2():
    params = __create_params_ones_kx1([-50], [10])
    x, y = params.simulate(100)
    kf = KalmanFilter()
    kf.parameters = params
    kf.set_Y(y)
    kf.filter()
    ks = KalmanSmoother()
    ks.parameters = params
    ks.set_Y(y)
    ks.smooth()
    assert (np.abs(np.mean(ks.Xp()) - -50) <= 0.1 * 50), "Failed simulation: mean(X pred) != true mean"
    assert (np.abs(np.mean(ks.Xf()) - -50) <= 0.1 * 50), "Failed simulation: mean(X filter) != true mean"
    assert (np.abs(np.mean(ks.Xs()) - -50) <= 0.1 * 50), "Failed simulation: mean(X smooth) != true mean"
    #assert np.round(np.std(ks.Xp()), 2) >= np.round(np.std(ks.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)"
    #assert np.round(np.std(ks.Xf()), 2) >= np.round(np.std(ks.Xs()), 2), "Failed simulation: std(X filter) < std(X smooth)"
    
def kalman_smoother(Y, F, H, Q, R, X0, P0):
    kf = KalmanSmoother()
    kf.set_F(F)
    kf.set_H(H)
    kf.set_Q(Q)
    kf.set_R(R)
    kf.set_X0(X0)
    kf.set_P0(P0)
    kf.set_Y(Y)
    kf.set_obs_dim(_nrows(Y))
    kf.set_latent_dim(_nrows(X0))
    kf.smooth()
    return kf

def kalman_smoother_from_parameters(Y, params):
    kf = KalmanSmoother()
    kf.parameters = params
    kf.set_Y(Y)
    kf.set_obs_dim(_nrows(Y))
    kf.set_latent_dim(_nrows(params.X0))
    kf.smooth()
    return kf

def test_smoother_3():
    params = __create_params_ones_kx1([-50], [10])
    x, y = params.simulate(100)
    kf = kalman_smoother(y, params.F, params.H, params.Q, params.R, params.X0, params.P0)
    assert (np.abs(np.mean(kf.Xp()) - -50) <= 0.1 * 50), "Failed simulation: mean(X pred) != true mean"
    assert (np.abs(np.mean(kf.Xf()) - -50) <= 0.1 * 50), "Failed simulation: mean(X filter) != true mean"
    assert (np.abs(np.mean(kf.Xs()) - -50) <= 0.1 * 50), "Failed simulation: mean(X smooth) != true mean"
    #assert np.round(np.std(kf.Xp()), 2) >= np.round(np.std(kf.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)"
    #assert np.round(np.std(kf.Xf()), 2) >= np.round(np.std(kf.Xs()), 2), "Failed simulation: std(X filter) < std(X smooth)"
    kf = kalman_smoother_from_parameters(y, params)
    assert (np.abs(np.mean(kf.Xp()) - -50) <= 0.1 * 50), "Failed simulation: mean(X pred) != true mean"
    assert (np.abs(np.mean(kf.Xf()) - -50) <= 0.1 * 50), "Failed simulation: mean(X filter) != true mean"
    assert (np.abs(np.mean(kf.Xs()) - -50) <= 0.1 * 50), "Failed simulation: mean(X smooth) != true mean"
    #assert np.round(np.std(kf.Xp()), 2) >= np.round(np.std(kf.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)"
    #assert np.round(np.std(kf.Xf()), 2) >= np.round(np.std(kf.Xs()), 2), "Failed simulation: std(X filter) < std(X smooth)"


#####################################################################
# EM SSM Estimator
#####################################################################

class ExpectationMaximizationEstimator:
    def __init__(self):
        self.parameters = None
        self.Y = None
        self.estimate_F = True
        self.estimate_H = True
        self.estimate_Q = True
        self.estimate_R = True
        self.estimate_X0 = True
        self.estimate_P0 = True
        self.loglikelihood_record = []
        self.max_iterations = 10
        self.min_iterations = 10
        self.iteration_tolerance = 0.01
    
    #
    # Trick in C++ for optional references
    # static double _dummy_foobar;
    # void foo(double &bar, double &foobar = _dummy_foobar)
    #
    def set_parameters(self, Y, parameters=None, est_F=True, est_H=True, est_Q=True, est_R=True, est_X0=True, est_P0=True, latent_dim=None):
        if parameters is not None:
            self.parameters = parameters
        else:
            if latent_dim is None:
                raise ValueError("latent_dim unset!")
            self.parameters = SSMParameters()
            self.parameters.o = _ncols(Y)
            self.parameters.l = latent_dim
            self.parameters.random_initialize()
        self.Y = Y
        self.estimate_F = est_F
        self.estimate_H = est_H
        self.estimate_Q = est_Q
        self.estimate_R = est_R
        self.estimate_X0 = est_X0
        self.estimate_P0 = est_P0

    def estimation_iteration(self):
        ks = kalman_smoother_from_parameters(self.Y, self.parameters)
        self.loglikelihood_record.append(ks.loglikelihood())
        T = _ncols(ks.Y())
        L = self.parameters.latent_signal_dimension()
        O = self.parameters.observable_signal_dimension()
        P = _zero_cube(L, L, T)
        ACF = _zero_cube(L, L, T - 1)

        for i in range(T):
            _set_slice(P, i, _slice(ks.Ps(), i) + _dot(_col(ks.Xs(), i), _t(_col(ks.Xs(), i))))
            if i < T - 1:
                _set_slice(ACF, i, _slice(ks.Ps(), i) + _dot(_col(ks.Xs(), i + 1), _t(_col(ks.Xs(), i))))
        
        if self.estimate_H:
            self.parameters.H = _inv(_sum_cube(P))
            H1 = _zero_matrix(L, L)
            for t in range(T):
                H1 += _dot(_col(ks.Y(), t), _t(_col(ks.Xs(), t)))
            self.parameters.H = _dot(H1, self.parameters.H)
            # Fix math rounding errors
            _set_diag_values_positive(self.parameters.H)
        if self.estimate_R:
            self.parameters.R = _zero_matrix(O, O)
            for t in range(T):
                self.parameters.R += _dot(_col(ks.Y(), t) * _t(_col(ks.Y(), t))) - _dot(self.parameters.H, _col(ks.Xs(), t), _t(_col(ks.Y(), t)))
            self.parameters.R /= T
            _set_diag_values_positive(self.parameters.R)
        if self.estimate_F:
            self.parameters.F = _dot(_sum_cube(ACF), _inv(_sum_cube(_head_slices(P))))
        if self.estimate_Q:
            self.parameters.Q = _sum_cube(_tail_slices(P)) - _dot(self.parameters.F, _t(_sum_cube(ACF))) / (T - 1)
            _set_diag_values_positive(self.parameters.Q)
        if self.estimate_X0:
            self.parameters.X0 = _col(ks.Xs(), 0)
        if self.estimate_P0:
            self.parameters.P0 = _slice(ks.Ps(), 0)

    def estimate_parameters(self):
        self.estimation_iteration()
        for i in range(self.max_iterations):
            self.estimation_iteration()
            unsufficient_increment = self.loglikelihood_record[-1] - self.loglikelihood_record[-2] <= self.iteration_tolerance
            if unsufficient_increment and i > self.min_iterations:
                break
        ks = kalman_smoother_from_parameters(self.Y, self.parameters)
        self.loglikelihood_record.append(ks.loglikelihood())

def test_expectation_maximization_1():
    params_orig = __create_params_ones_kx1([-50], [10])
    params = __create_params_ones_kx1([-50], [10])
    params.X0[0, 0] += 0.1
    params.H[0, 0] -= 0.1
    params.show()
    x, y = params.simulate(100)
    kf = ExpectationMaximizationEstimator()
    kf.set_parameters(y, params)
    kf.estimate_parameters()
    kf.parameters.show()
    params.show()
    params_orig.show()
    #assert np.round(np.std(kf.Xp()), 2) >= np.round(np.std(kf.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)"
    #assert np.round(np.std(kf.Xf()), 2) >= np.round(np.std(kf.Xs()), 2), "Failed simulation: std(X filter) < std(X smooth)"


#####################################################################
# Heuristic SSM Estimator
#####################################################################


if __name__ == "__main__":
    import pytest
    #pytest.main(sys.argv[0])
    test_expectation_maximization_1()

# For testing run
# pip install pytest
# or
# py.test kalman.py
