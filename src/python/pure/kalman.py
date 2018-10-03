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
    return np.exp(-0.5 * _dot((x - mean).T, _inv(cov), (x - mean))) / np.sqrt(2 * np.pi * np.linalg.det(cov))

def _mvn_logprobability(x, mean, cov):
    return (-0.5 * _dot((x - mean).T, _inv(cov), (x - mean))) - 0.5 * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(cov))

def test_mvn_probability():
    xs = np.linspace(-5, 5, 101)
    dxs = xs[1] - xs[0]
    cdf = 0
    for x in xs:
        cdf += _mvn_probability(np.array([[x]]), mean=np.array([[0]]), cov=np.array([[1]])) * dxs
    assert np.abs(cdf - 1) < 1e-3, "Bad integration!"
    cdf = 0
    for x in xs:
        cdf += _mvn_probability(np.array([[x]]), mean=np.array([[0]]), cov=np.array([[1]]))
    xs = np.linspace(-5, 5, 41)
    dxs = xs[1] - xs[0]
    cdf = 0
    dxs = (dxs * dxs * 0.4)
    for x in xs:
        for y in xs:
            cdf += _mvn_probability(np.array([[x], [y]]), mean=np.array([[0], [0]]), cov=np.array([[1, 0], [0, 1]])) * dxs
            #print(
            #"    ", dxs ** 2, _mvn_probability(np.array([[x], [y]]), mean=np.array([[0], [0]]), cov=np.array([[1, 0], [0, 1]])) * dxs * dxs
            #, file=sys.stderr)
    assert np.abs(cdf - 1) < 1e-2, "Bad integration!"

def test_mvn_logprobability():
    xs = np.linspace(-5, 5, 101)
    dxs = xs[1] - xs[0]
    cdf = 0
    for x in xs:
        cdf += np.exp(_mvn_logprobability(x, mean=np.array([0]), cov=np.array([[1]]))) * dxs
    assert np.abs(cdf - 1) < 1e-3, "Bad integration!"
    xs = np.linspace(-5, 5, 41)
    dxs = xs[1] - xs[0]
    dxs = (dxs * dxs * 0.4)
    cdf = 0
    for x in xs:
        for y in xs:
            cdf += np.exp(_mvn_logprobability(np.array([[x], [y]]), mean=np.array([[0], [0]]), cov=np.array([[1, 0], [0, 1]]))) * dxs
    assert np.abs(cdf - 1) < 1e-2, "Bad integration!"

def _mvn_sample(mean, cov):
    return scipy.stats.multivariate_normal.rvs(mean=mean.ravel(), cov=cov).reshape(mean.shape)

#####################################################################
# Math cross-platform functions
#####################################################################

def _create_noised_ones(L, M):
    return np.ones((L, M), dtype="f") + 0.05 * np.random.random((L, M))

def _create_noised_zeros(L, M):
    return np.zeros((L, M), dtype="f") + 0.05 * np.random.random((L, M))

def _create_noised_diag(L, M):
    return np.eye(L, M, dtype="f") + 0.05 * np.random.random((L, M))

def test_create_noised_ones():
    m = _create_noised_ones(1, 1)
    assert np.abs(m[0][0] - 1) < 0.1
    m = _create_noised_ones(10, 10)
    for i in range(10):
        for j in range(10):
            assert np.abs(m[i][j] - 1) < 0.1

def test_create_noised_zeros():
    m = _create_noised_zeros(1, 1)
    assert np.abs(m[0][0] - 0) < 0.1
    m = _create_noised_zeros(10, 10)
    for i in range(10):
        for j in range(10):
            assert np.abs(m[i][j] - 0) < 0.1

def test_create_noised_diag():
    m = _create_noised_diag(1, 1)
    assert np.abs(m[0][0] - 1) < 0.1
    m = _create_noised_diag(10, 10)
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

def _subsample(Y, sample_size):
    if sample_size >= Y.shape[1]: return Y
    index = np.arange(Y.shape[1])
    np.random.shuffle(index)
    index = index[: sample_size]
    return Y[:, index]

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
        self.obs_dim = -1
        self.lat_dim = -1

    def set_dimensions(self, obs_dim, lat_dim):
        self.obs_dim = obs_dim
        self.lat_dim = lat_dim
    
    def latent_signal_dimension(self):
        return self.lat_dim

    def observable_signal_dimension(self):
        return self.obs_dim
    
    def show(self):
        print(self)
        print(" Transition matrix: ", self.F)
        print(" Observation matrix: ", self.H)
        print(" Latent var-covar matrix: ", self.Q)
        print(" Observation var-covar matrix: ", self.R)
        print(" Initial latent signal: ", self.X0)
        print(" Initial latent var-covar matrix: ", self.P0)

    def random_initialize(self, init_F=True, init_H=True, init_Q=True, init_R=True, init_X0=True, init_P0=True):
        if self.lat_dim < 0:
            raise ValueError("Latent signal dimension is unset!")
        if self.obs_dim < 0:
            raise ValueError("Observable signal dimension is unset!")
        if init_F: self.F = _create_noised_ones(self.lat_dim, self.lat_dim)
        if init_Q: self.Q = _create_noised_diag(self.lat_dim, self.lat_dim)
        if init_X0: self.X0 = _create_noised_ones(self.lat_dim, 1)
        if init_P0: self.P0 = _create_noised_diag(self.lat_dim, self.lat_dim)
        if init_H: self.H = _create_noised_ones(self.obs_dim, self.lat_dim)
        if init_R: self.R = _create_noised_diag(self.obs_dim, self.obs_dim)
    
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
    
    def copy(self):
        p = SSMParameters()
        p.F = self.F.copy().astype("f")
        p.H = self.H.copy().astype("f")
        p.Q = self.Q.copy().astype("f")
        p.R = self.R.copy().astype("f")
        p.X0 = self.X0.copy().astype("f")
        p.P0 = self.P0.copy().astype("f")
        p.obs_dim = self.obs_dim
        p.lat_dim = self.lat_dim
        return p
    
    def copy_from(self, p):
        self.F = p.F.copy().astype("f")
        self.H = p.H.copy().astype("f")
        self.Q = p.Q.copy().astype("f")
        self.R = p.R.copy().astype("f")
        self.X0 = p.X0.copy().astype("f")
        self.P0 = p.P0.copy().astype("f")
        self.obs_dim = p.obs_dim
        self.lat_dim = p.lat_dim
        return p


def _create_params_ones_kx1(M, K=[100.0]):
    K = np.array([K]).T
    params = SSMParameters()
    params.F = np.array([[1-1e-10]])
    params.H = K
    params.Q = np.array([[0.01]])
    params.R = 0.01 * _create_noised_diag(_nrows(K), _nrows(K))
    params.X0 = np.array([M])
    params.P0 = np.array([[0.01]])
    params.set_dimensions(len(K), 1)
    return params

def test_simulations_params():
    params = _create_params_ones_kx1(-50, [10])
    _, y = params.simulate(100)
    assert np.abs(np.round(np.mean(_row(y, 0))) - -500)/100 < 0.1 * 500, "Failed simulation"

    params = _create_params_ones_kx1(2, [-100, 100])
    _, y = params.simulate(1)
    assert np.abs(np.round(np.mean(_row(y, 0))) - -200) < 0.1 * 200, "Failed simulation"
    assert np.abs(np.round(np.mean(_row(y, 1))) - 200) < 0.1 * 200, "Failed simulation"
    
    _, y = params.simulate(100)
    assert np.abs(np.round(np.mean(_row(y, 0))) - -200)/100 < 0.1 * 200, "Failed simulation"
    assert np.abs(np.round(np.mean(_row(y, 1))) - 200)/100 < 0.1 * 200, "Failed simulation"
    

class SSMEstimated:
    def __init__(self):
        self.X = None
        self.Y = None
        self.P = None
        self.ACV1 = None
    
    def signal(self): return self.X

    def variance(self): return self.P
    
    def autocovariance(self): return self.ACV1
    
    def init(self, dimX, dimY, length, fill_ACV=False):
        self.X = _zero_matrix(dimX, length)
        self.Y = _zero_matrix(dimY, length)
        self.P = _zero_cube(dimX, dimX, length)
        if fill_ACV:
            self.ACV1 = _zero_cube(dimX, dimX, length - 1)


def _predict_expected_ssm(H, Xpred):
    Ypred = _zero_matrix(_nrows(H), _ncols(Xpred))
    for t in range(_ncols(Xpred)):
        _set_col(Ypred, t, _dot(H, _col(Xpred, t)))
    return Ypred

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
    
    def obs_dim(self): return self.parameters.obs_dim
    
    def set_obs_dim(self, v): self.parameters.obs_dim = v
    
    def lat_dim(self): return self.parameters.lat_dim
    
    def set_lat_dim(self, v): self.parameters.lat_dim = v

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
    
    def Yf(self): return self.filtered_estimates.Y

    def Xp(self): return self.predicted_estimates.X
        
    def Yp(self): return self.predicted_estimates.Y

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
        if self.lat_dim() == 0:
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
        self.filtered_estimates.init(self.lat_dim(), self.obs_dim(), self.T())
        self.predicted_estimates.init(self.lat_dim(), self.obs_dim(), self.T())
        
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
        #
        self.predicted_estimates.Y = _predict_expected_ssm(self.H(), self.predicted_estimates.X)
        self.filtered_estimates.Y = _predict_expected_ssm(self.H(), self.filtered_estimates.X)


def test_filter_1():
    params = _create_params_ones_kx1([-50], [10])
    x, y = params.simulate(100)
    kf = KalmanFilter()
    kf.set_Y(y)
    kf.parameters = params
    kf.filter()
    assert (np.abs(np.mean(kf.Xp()) - -50) <= 0.1 * 50), "Failed simulation: mean(X pred) != true mean"
    assert (np.abs(np.mean(kf.Xf()) - -50) <= 0.1 * 50), "Failed simulation: mean(X filter) != true mean"
    #assert np.round(np.std(kf.Xp()), 2) >= np.round(np.std(kf.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)"


# [[export]]
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
    kf.set_lat_dim(_nrows(X0))
    kf.filter()
    return kf

def kalman_filter_from_parameters(Y, params):
    kf = KalmanFilter()
    kf.parameters = params
    kf.set_Y(Y)
    kf.set_obs_dim(_nrows(Y))
    kf.set_lat_dim(_nrows(params.X0))
    kf.filter()
    return kf

# [[export]]
def kalman_filter_results(kf):
    return (
        kf.Xp(), kf.Pp(), kf.Yp(),
        kf.Xf(), kf.Pf(), kf.Yf(),
    )

# [[export]]
def kalman_filter_parameters(kf):
    kf = KalmanFilter()
    return (
        kf.F(), kf.H(), kf.Q(), 
        kf.R(), kf.X0(), kf.P0(),
    )


def test_filter_2():
    params = _create_params_ones_kx1([-50], [10])
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

    def Ys(self): return self.smoothed_estimates.Y

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
        
        self.smoothed_estimates.init(self.lat_dim(), self.obs_dim(), self.T(), True)
        
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
        #
        self.smoothed_estimates.Y = _predict_expected_ssm(self.H(), self.smoothed_estimates.X)

def test_smoother_1():
    params = _create_params_ones_kx1([-50], [10])
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
    params = _create_params_ones_kx1([-50], [10])
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
    
# [[export]]
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
    kf.set_lat_dim(_nrows(X0))
    kf.smooth()
    return kf

def kalman_smoother_from_parameters(Y, params):
    kf = KalmanSmoother()
    kf.parameters = params
    kf.set_Y(Y)
    kf.set_obs_dim(_nrows(Y))
    kf.set_lat_dim(_nrows(params.X0))
    kf.smooth()
    return kf

# [[export]]
def kalman_smoother_results(kf):
    return (
        kf.Xp(), kf.Pp(), kf.Yp(),
        kf.Xf(), kf.Pf(), kf.Yf(),
        kf.Xs(), kf.Ps(), kf.Ys(),
    )

# [[export]]
def kalman_smoother_parameters(kf):
    kf = KalmanSmoother()
    return (
        kf.F(), kf.H(), kf.Q(), 
        kf.R(), kf.X0(), kf.P0(),
    )

def test_smoother_3():
    params = _create_params_ones_kx1([-50], [10])
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
        self.min_iterations = 1
        self.min_improvement = 0.01
    
    #
    # Trick in C++ for optional references
    # static double _dummy_foobar;
    # void foo(double &bar, double &foobar = _dummy_foobar)
    #
    def set_parameters(self, Y, parameters=None, est_F=True, est_H=True, est_Q=True, est_R=True, est_X0=True, est_P0=True, lat_dim=None):
        if parameters is not None:
            self.parameters = parameters
        else:
            if lat_dim is None:
                raise ValueError("lat_dim unset!")
            self.parameters = SSMParameters()
            self.parameters.obs_dim = _ncols(Y)
            self.parameters.lat_dim = lat_dim
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
            unsufficient_increment = self.loglikelihood_record[-1] - self.loglikelihood_record[-2] <= self.min_improvement
            if unsufficient_increment and i > self.min_iterations:
                break
        ks = kalman_smoother_from_parameters(self.Y, self.parameters)
        self.loglikelihood_record.append(ks.loglikelihood())
    
    def smoother(self):
        return kalman_smoother_from_parameters(self.Y, self.parameters)

def test_expectation_maximization_1():
    params_orig = _create_params_ones_kx1([-50], [10])
    params = _create_params_ones_kx1([-50], [10])
    params.X0[0, 0] += 1
    params.H[0, 0] -= 0.3
    params.F[0, 0] -= 0.1
    #params.show()
    x, y = params.simulate(100)
    kf = ExpectationMaximizationEstimator()
    kf.set_parameters(y, params)
    kf.estimate_parameters()
    #kf.parameters.show()
    #params.show()
    #params_orig.show()
    assert (np.abs(np.mean(params.X0) - -50) <= 0.15 * 50), "Failed simulation: mean(X0 pred) != true mean"
    assert (np.abs(np.mean(params.F) - 1) <= 0.15 * 1), "Failed simulation: mean(F pred) != true mean"
    assert (np.abs(np.mean(params.H) - 10) <= 0.15 * 10), "Failed simulation: mean(H pred) != true mean"
    assert (np.abs(np.mean(params.X0) - np.mean(params_orig.X0)) <= 0.15 * 50), "Failed simulation: mean(X0 est) !~= mean(X0 pred)"
    assert (np.abs(np.mean(params.F) - np.mean(params_orig.F)) <= 0.15 * 1), "Failed simulation: mean(F est) !~= mean(F orig)"
    assert (np.abs(np.mean(params.H) - np.mean(params_orig.H)) <= 0.15 * 10), "Failed simulation: mean(H est) !~= mean(H orig)"
    assert (np.abs(np.mean(params.X0) - np.mean(params_orig.X0)) > 0), "Failed simulation: mean(X0 est) == mean(X0 pred) (it was copied?)"
    assert (np.abs(np.mean(params.F) - np.mean(params_orig.F)) > 0), "Failed simulation: mean(F est) == mean(F orig) (it was copied?)"
    assert (np.abs(np.mean(params.H) - np.mean(params_orig.H)) > 0), "Failed simulation: mean(H est) == mean(H orig) (it was copied?)"
    #assert np.round(np.std(kf.Xp()), 2) >= np.round(np.std(kf.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)"
    #assert np.round(np.std(kf.Xp()), 2) >= np.round(np.std(kf.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)"
    #assert np.round(np.std(kf.Xf()), 2) >= np.round(np.std(kf.Xs()), 2), "Failed simulation: std(X filter) < std(X smooth)"

def test_expectation_maximization_2():
    params_orig = _create_params_ones_kx1([-50], [10])
    params = _create_params_ones_kx1([-50], [10])
    #params.X0[0, 0] += 1
    #params.H[0, 0] -= 0.3
    #params.F[0, 0] -= 0.1
    #params.show()
    x, y = params.simulate(100)
    kf = ExpectationMaximizationEstimator()
    kf.set_parameters(y, params)
    kf.estimate_parameters()
    #kf.parameters.show()
    #params.show()
    #params_orig.show()
    assert (np.abs(np.mean(params.X0) - -50) <= 0.15 * 50), "Failed simulation: mean(X0 pred) != true mean"
    assert (np.abs(np.mean(params.F) - 1) <= 0.15 * 1), "Failed simulation: mean(F pred) != true mean"
    assert (np.abs(np.mean(params.H) - 10) <= 0.15 * 10), "Failed simulation: mean(H pred) != true mean"
    assert (np.abs(np.mean(params.X0) - np.mean(params_orig.X0)) <= 0.15 * 50), "Failed simulation: mean(X0 est) !~= mean(X0 pred)"
    assert (np.abs(np.mean(params.F) - np.mean(params_orig.F)) <= 0.15 * 1), "Failed simulation: mean(F est) !~= mean(F orig)"
    assert (np.abs(np.mean(params.H) - np.mean(params_orig.H)) <= 0.15 * 10), "Failed simulation: mean(H est) !~= mean(H orig)"
    assert (np.abs(np.mean(params.X0) - np.mean(params_orig.X0)) > 0), "Failed simulation: mean(X0 est) == mean(X0 pred) (it was copied?)"
    assert (np.abs(np.mean(params.F) - np.mean(params_orig.F)) > 0), "Failed simulation: mean(F est) == mean(F orig) (it was copied?)"
    assert (np.abs(np.mean(params.H) - np.mean(params_orig.H)) > 0), "Failed simulation: mean(H est) == mean(H orig) (it was copied?)"
    #assert np.round(np.std(kf.Xp()), 2) >= np.round(np.std(kf.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)"
    #assert np.round(np.std(kf.Xp()), 2) >= np.round(np.std(kf.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)"
    #assert np.round(np.std(kf.Xf()), 2) >= np.round(np.std(kf.Xs()), 2), "Failed simulation: std(X filter) < std(X smooth)"

# [[export]]
def estimate_using_em(Y, estimates="", F0=None, H0=None, Q0=None, R0=None, X00=None, P00=None, min_iterations=1, max_iterations=10, min_improvement=0.01, lat_dim=None):
    estimator = ExpectationMaximizationEstimator()
    estimator.Y = Y
    estimator.estimate_F = "F" in estimates
    estimator.estimate_H = "H" in estimates
    estimator.estimate_Q = "Q" in estimates
    estimator.estimate_R = "R" in estimates
    estimator.estimate_X0 = "X0" in estimates
    estimator.estimate_P0 = "P0" in estimates
    estimator.parameters = SSMParameters()
    estimator.parameters.F = F0
    if F0 is not None:
        estimator.parameters.lat_dim = _nrows(F0)
    estimator.parameters.H = H0
    if H0 is not None:
        estimator.parameters.lat_dim = _ncols(H0)
    estimator.parameters.Q = Q0
    if Q0 is not None:
        estimator.parameters.lat_dim = _ncols(Q0)
    estimator.parameters.R = R0
    estimator.parameters.X0 = X00
    if X00 is not None:
        estimator.parameters.lat_dim = _nrows(X00)
    estimator.parameters.P0 = P00
    if P00 is not None:
        estimator.parameters.lat_dim = _ncols(P00)
    if lat_dim is not None:
        estimator.parameters.lat_dim = lat_dim
    estimator.parameters.obs_dim = _nrows(Y)
    estimator.parameters.obs_dim = _nrows(Y)
    estimator.parameters.random_initialize(F0 is None, H0 is None, Q0 is None, R0 is None, X00 is None, P00 is None)
    estimator.estimate_parameters()
    ks = estimator.smoother()
    ks.smooth()
    return ks, estimator.loglikelihood_record

def test_expectation_maximization_3():
    params = _create_params_ones_kx1([-50], [10])
    x, y = params.simulate(100)
    ks, records = estimate_using_em(y,
        estimates="F H Q R X0 P0",
        F0=params.F, H0=params.H, Q0=params.Q, R0=params.R, X00=params.X0, P00=params.P0,
        min_iterations=1, max_iterations=10, min_improvement=0.01, lat_dim=None)
    neoparams = ks.parameters
    print(records)
    neoparams.show()
    params.show()
    #params_orig.show()
    assert (np.abs(np.mean(neoparams.X0) - -50) <= 0.15 * 50), "Failed simulation: mean(X0 pred) != true mean"
    assert (np.abs(np.mean(neoparams.F) - 1) <= 0.15 * 1), "Failed simulation: mean(F pred) != true mean"
    assert (np.abs(np.mean(neoparams.H) - 10) <= 0.15 * 10), "Failed simulation: mean(H pred) != true mean"
    
#####################################################################
# Heuristic SSM Estimator
#####################################################################

class HeuristicEstimatorParticle:
    def __init__(self):
        self.params = SSMParameters()
        self.metric = -1e100
        self.best_params = SSMParameters()
        self.best_metric = -1e150
    
    # Assume that any param not null is fixed
    def init(self, obs_dim, lat_dim, F0=None, H0=None, Q0=None, R0=None, X00=None, P00=None):
        #self.params = SSMParameters()
        self.params.F = F0.astype("f")
        #if F0 is not None:
        #    self.params.lat_dim = _nrows(F0)
        self.params.H = H0.astype("f")
        #if H0 is not None:
        #    self.params.lat_dim = _ncols(H0)
        self.params.Q = Q0.astype("f")
        #if Q0 is not None:
        #    self.params.lat_dim = _ncols(Q0)
        self.params.R = R0.astype("f")
        self.params.X0 = X00.astype("f")
        #if X00 is not None:
        #    self.params.lat_dim = _nrows(X00)
        self.params.P0 = P00.astype("f")
        #if P00 is not None:
        #    self.params.lat_dim = _ncols(P00)
        #if lat_dim is not None:
        #    self.params.lat_dim = lat_dim
        self.params.lat_dim = lat_dim
        self.params.obs_dim = obs_dim
        self.params.random_initialize(F0 is None, H0 is None, Q0 is None, R0 is None, X00 is None, P00 is None)
        self.best_params = self.params.copy()
    
    def init_with_parameters(self, obs_dim, parameters=None, est_F=True, est_H=True, est_Q=True, est_R=True, est_X0=True, est_P0=True, lat_dim=None):
        if parameters is not None:
            #self.params = parameters.copy()
            self.params.copy_from(parameters)
        else:
            if lat_dim is None:
                raise ValueError("lat_dim unset!")
            #self.params = SSMParameters()
            self.params.lat_dim = lat_dim
            self.params.random_initialize()
        self.params.obs_dim = obs_dim
        self.params.random_initialize(est_F, est_H, est_Q, est_R, est_X0, est_P0)
        self.best_params = self.params.copy()

    def evaluate(self, Y):
        ks = kalman_smoother_from_parameters(Y, self.params)
        self.metric = ks.loglikelihood()
        if self.metric > self.best_metric:
            self.best_metric = self.metric
            self.params = self.best_params.copy()
        
    def move(self, best_particle, est_F=True, est_H=True, est_Q=True, est_R=True, est_X0=True, est_P0=True):
        move_to_self_best = 2 * np.random.uniform()
        move_to_global_best = 2 * np.random.uniform()
        if est_F:
            self.params.F += move_to_self_best * (self.best_params.F - self.params.F) + move_to_global_best * (best_particle.best_params.F - self.params.F)
        if est_H:
            self.params.H += move_to_self_best * (self.best_params.H - self.params.H) + move_to_global_best * (best_particle.best_params.H - self.params.H)
        if est_Q:
            self.params.Q += move_to_self_best * (self.best_params.Q - self.params.Q) + move_to_global_best * (best_particle.best_params.Q - self.params.Q)
            self.params.Q = 0.5 * (self.params.Q + _t(self.params.Q))
            _set_diag_values_positive(self.params.Q)
        if est_R:
            self.params.R += move_to_self_best * (self.best_params.R - self.params.R) + move_to_global_best * (best_particle.best_params.R - self.params.R)
            self.params.R = 0.5 * (self.params.R + _t(self.params.R))
            _set_diag_values_positive(self.params.R)
        if est_X0:
            self.params.X0 += move_to_self_best * (self.best_params.X0 - self.params.X0) + move_to_global_best * (best_particle.best_params.X0 - self.params.X0)
        if est_P0:
            self.params.P0 += move_to_self_best * (self.best_params.P0 - self.params.P0) + move_to_global_best * (best_particle.best_params.P0 - self.params.P0)
            self.params.P0 = 0.5 * (self.params.P0 + _t(self.params.P0))
            _set_diag_values_positive(self.params.P0)
    
    def copy_best_from(self, other, force_copy=False):
        if other.best_metric > self.metric or force_copy:
            self.metric = other.best_metric
            self.best_metric = other.best_metric
            self.params.copy_from(other.best_params)
            self.best_params.copy_from(other.best_params)
    

class PurePSOHeuristicEstimator:
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
        self.max_iterations = 50
        self.min_iterations = 10
        self.min_improvement = 0.01
        self.sample_size = 30
        self.population_size = 20
        self.particles = []
        self.best_particle = None

    def set_parameters(self, Y, parameters=None, est_F=True, est_H=True, est_Q=True, est_R=True, est_X0=True, est_P0=True, lat_dim=None):
        #
        if parameters is not None:
            self.parameters = parameters
        else:
            self.parameters = SSMParameters()
            self.parameters.obs_dim = _nrows(Y)
            self.parameters.lat_dim = lat_dim
            if lat_dim is None:
                raise ValueError("lat_dim unset!")
            self.parameters.random_initialize(est_F, est_H, est_Q, est_R, est_X0, est_P0)
        #
        self.Y = Y
        self.sample_size = _ncols(Y)
        self.estimate_F = est_F
        self.estimate_H = est_H
        self.estimate_Q = est_Q
        self.estimate_R = est_R
        self.estimate_X0 = est_X0
        self.estimate_P0 = est_P0
        #
        self.best_particle = HeuristicEstimatorParticle()
        self.particles = []
        for i in range(self.population_size):
            self.particles.append(HeuristicEstimatorParticle())
            if i == 0:
                self.particles[i].init_with_parameters(_nrows(Y), parameters.copy(), False, False, False, False, False, False, parameters.lat_dim)
            else:
                self.particles[i].init_with_parameters(_nrows(Y), parameters.copy(), est_F, est_H, est_Q, est_R, est_X0, est_P0, lat_dim)
            self.particles[i].evaluate(_subsample(self.Y, self.sample_size))
            self.best_particle.copy_best_from(self.particles[i], True)
            print(" **  ", self.particles[i].metric)
            self.particles[i].params.show()
        #
        self.parameters.copy_from(self.best_particle.best_params)
        #

    def estimation_iteration_heuristic(self):
        for i in range(self.population_size):
            #self.particles[i].evaluate(self.Y)
            self.particles[i].evaluate(_subsample(self.Y, self.sample_size))
            self.best_particle.copy_best_from(self.particles[i])
            self.particles[i].move(self.best_particle, self.estimate_F, self.estimate_H, self.estimate_Q, self.estimate_R, self.estimate_X0, self.estimate_P0)
        self.loglikelihood_record.append(self.best_particle.best_metric)
        self.parameters.copy_from(self.best_particle.best_params)
        print(" >>>> ", self.best_particle.metric)
        self.best_particle.params.show()
    
    def estimate_parameters(self):
        self.estimation_iteration_heuristic()
        for i in range(self.max_iterations):
            self.estimation_iteration_heuristic()
            unsufficient_increment = self.loglikelihood_record[-1] - self.loglikelihood_record[-2] <= self.min_improvement
            if unsufficient_increment and i > self.min_iterations:
                break
        ks = kalman_smoother_from_parameters(self.Y, self.parameters)
        self.loglikelihood_record.append(ks.loglikelihood())
    
    def smoother(self):
        return kalman_smoother_from_parameters(self.Y, self.parameters)


def test_pure_pso_1():
    params_orig = _create_params_ones_kx1([-50], [10])
    params = _create_params_ones_kx1([-50], [10])
    params.X0[0, 0] += 1
    params.H[0, 0] -= 0.3
    params.F[0, 0] -= 0.1
    #params.show()
    x, y = params.simulate(100)
    kf = PurePSOHeuristicEstimator()
    kf.set_parameters(y, params)
    kf.estimate_parameters()
    kf.parameters.show()
    print(kf.loglikelihood_record)
    #params.show()
    #params_orig.show()
    assert (np.abs(np.mean(params.X0) - -50) <= 0.15 * 50), "Failed simulation: mean(X0 pred) != true mean"
    assert (np.abs(np.mean(params.F) - 1) <= 0.15 * 1), "Failed simulation: mean(F pred) != true mean"
    assert (np.abs(np.mean(params.H) - 10) <= 0.15 * 10), "Failed simulation: mean(H pred) != true mean"
    assert (np.abs(np.mean(params.X0) - np.mean(params_orig.X0)) <= 0.15 * 50), "Failed simulation: mean(X0 est) !~= mean(X0 pred)"
    assert (np.abs(np.mean(params.F) - np.mean(params_orig.F)) <= 0.15 * 1), "Failed simulation: mean(F est) !~= mean(F orig)"
    assert (np.abs(np.mean(params.H) - np.mean(params_orig.H)) <= 0.15 * 10), "Failed simulation: mean(H est) !~= mean(H orig)"
    assert (np.abs(np.mean(params.X0) - np.mean(params_orig.X0)) > 0), "Failed simulation: mean(X0 est) == mean(X0 pred) (it was copied?)"
    assert (np.abs(np.mean(params.F) - np.mean(params_orig.F)) > 0), "Failed simulation: mean(F est) == mean(F orig) (it was copied?)"
    assert (np.abs(np.mean(params.H) - np.mean(params_orig.H)) > 0), "Failed simulation: mean(H est) == mean(H orig) (it was copied?)"
    #assert np.round(np.std(kf.Xp()), 2) >= np.round(np.std(kf.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)"
    #assert np.round(np.std(kf.Xp()), 2) >= np.round(np.std(kf.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)"
    #assert np.round(np.std(kf.Xf()), 2) >= np.round(np.std(kf.Xs()), 2), "Failed simulation: std(X filter) < std(X smooth)"

def test_pure_pso_2():
    params_orig = _create_params_ones_kx1([-50], [10])
    params = _create_params_ones_kx1([-50], [10])
    #params.X0[0, 0] += 1
    #params.H[0, 0] -= 0.3
    #params.F[0, 0] -= 0.1
    #params.show()
    x, y = params.simulate(100)
    kf = PurePSOHeuristicEstimator()
    kf.set_parameters(y, params)
    kf.estimate_parameters()
    #kf.parameters.show()
    #params.show()
    #params_orig.show()
    assert (np.abs(np.mean(params.X0) - -50) <= 0.15 * 50), "Failed simulation: mean(X0 pred) != true mean"
    assert (np.abs(np.mean(params.F) - 1) <= 0.15 * 1), "Failed simulation: mean(F pred) != true mean"
    assert (np.abs(np.mean(params.H) - 10) <= 0.15 * 10), "Failed simulation: mean(H pred) != true mean"
    assert (np.abs(np.mean(params.X0) - np.mean(params_orig.X0)) <= 0.15 * 50), "Failed simulation: mean(X0 est) !~= mean(X0 pred)"
    assert (np.abs(np.mean(params.F) - np.mean(params_orig.F)) <= 0.15 * 1), "Failed simulation: mean(F est) !~= mean(F orig)"
    assert (np.abs(np.mean(params.H) - np.mean(params_orig.H)) <= 0.15 * 10), "Failed simulation: mean(H est) !~= mean(H orig)"
    assert (np.abs(np.mean(params.X0) - np.mean(params_orig.X0)) > 0), "Failed simulation: mean(X0 est) == mean(X0 pred) (it was copied?)"
    assert (np.abs(np.mean(params.F) - np.mean(params_orig.F)) > 0), "Failed simulation: mean(F est) == mean(F orig) (it was copied?)"
    assert (np.abs(np.mean(params.H) - np.mean(params_orig.H)) > 0), "Failed simulation: mean(H est) == mean(H orig) (it was copied?)"
    #assert np.round(np.std(kf.Xp()), 2) >= np.round(np.std(kf.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)"
    #assert np.round(np.std(kf.Xp()), 2) >= np.round(np.std(kf.Xf()), 2), "Failed simulation: std(X pred) < std(X filter)"
    #assert np.round(np.std(kf.Xf()), 2) >= np.round(np.std(kf.Xs()), 2), "Failed simulation: std(X filter) < std(X smooth)"


if __name__ == "__main__":
    import pytest
    #pytest.main(sys.argv[0])
    test_expectation_maximization_3()

# For testing run
# pip install pytest
# or
# py.test kalman.py
