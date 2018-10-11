import os
import sys
import numpy as np
import scipy.stats

# This file was created with the idea to be
# a mockup class hierarchy for a C++ implementation
# Therefore, it is not totally Pythonic.
# Sorry if you expected other thing
# The same tests will be translated in C++

resourceParams = {"trace.intermediates": False,}

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
    return np.asscalar((-0.5 * _dot((x - mean).T, _inv(cov), (x - mean))) - 0.5 * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(cov)))

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

def _covariance_matrix_estimation(X): # unbiased estimation
    #Q = -1 * _dot(_sum_cols(X), _t(_sum_cols(X))) * (_ncols(X) - 1) / (_ncols(X) * _ncols(X))
    Q = -1 * _dot(_sum_cols(X), _t(_sum_cols(X))) / _ncols(X)
    for t in range(_ncols(X)):
        Q += _dot(_col(X, t), _t(_col(X, t)))
    Q /= (_ncols(X) - 1)
    if np.isscalar(Q):
        Q = np.array([[Q]])
    return Q

def test_covariance_matrix_estimation():
    X = np.array([[1, -2, 4], [1, 5, 3]])
    Y = _covariance_matrix_estimation(X)
    assert _ncols(Y) == 2
    assert _nrows(Y) == 2
    assert np.abs(Y[0, 0] -  9) < 0.01
    assert np.abs(Y[0, 1] - -3) < 0.01
    assert np.abs(Y[1, 0] - -3) < 0.01
    assert np.abs(Y[1, 1] -  4) < 0.01
    
    X = _zero_matrix(1, 4)
    X[0, 0] = 0
    X[0, 1] = 1
    X[0, 2] = 2
    X[0, 3] = 3
    Y = _covariance_matrix_estimation(X)
    assert _ncols(Y) == 1
    assert _nrows(Y) == 1
    assert np.abs(Y[0, 0] - 1.666) < 0.01


#####################################################################
# Math cross-platform functions
#####################################################################

def _create_noised_values(L, M):
    return np.random.randn(L, M)

def _create_noised_ones(L, M):
    return np.ones((L, M), dtype="f") + 0.05 * np.random.randn(L, M)

def _create_noised_zeros(L, M):
    return np.zeros((L, M), dtype="f") + 0.05 * np.random.randn(L, M)

def _create_noised_diag(L, M):
    return np.eye(L, M, dtype="f") + 0.05 * np.random.randn(L, M)

def test_create_noised_ones():
    m = _create_noised_ones(1, 1)
    assert np.abs(m[0][0] - 1) < 0.1
    m = _create_noised_ones(10, 10)
    for i in range(10):
        for j in range(10):
            assert np.abs(m[i][j] - 1) < 0.18

def test_create_noised_zeros():
    m = _create_noised_zeros(1, 1)
    assert np.abs(m[0][0] - 0) < 0.1
    m = _create_noised_zeros(10, 10)
    for i in range(10):
        for j in range(10):
            assert np.abs(m[i][j] - 0) < 0.18

def test_create_noised_diag():
    m = _create_noised_diag(1, 1)
    assert np.abs(m[0][0] - 1) < 0.1
    m = _create_noised_diag(10, 10)
    for i in range(10):
        for j in range(10):
            if i == j:
                assert np.abs(m[i][j] - 1) < 0.18
            else:
                assert np.abs(m[i][j] - 0) < 0.18

def _sum_slices(X):
    return np.sum(X, axis=2)

def test_sum_slices():
    X = np.zeros((2, 2, 4))
    X[:, :, 0] = [[0, 1], [2, 3]]
    X[:, :, 1] = [[0, 0], [0, 1]]
    X[:, :, 2] = [[1, 1], [0, -1]]
    X[:, :, 3] = [[2, 0], [2, 0]]
    Y = _sum_slices(X)
    assert Y[0][0] == 3
    assert Y[0][1] == 2
    assert Y[1][0] == 4
    assert Y[1][1] == 3
    X = np.zeros((1, 1, 3))
    X[:, :, 0] = [1]
    X[:, :, 1] = [1]
    X[:, :, 2] = [2]
    Y = _sum_slices(X)
    assert Y[0][0] == 4

def _sum_cols(X):
    return np.sum(X, axis=1).reshape(-1, 1)

def test_sum_cols():
    X = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    Y = _sum_cols(X)
    assert _ncols(Y) == 1
    assert _nrows(Y) == 2
    assert Y[0, 0] == 6
    assert Y[1, 0] == 22


def _set_diag_values_positive(X):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i, j] = np.abs(X[i, j])

def _subsample(Y, sample_size):
    if sample_size >= Y.shape[1]: return 0, Y
    i0 = np.random.random_integers(0, Y.shape[1] - sample_size - 1)
    return i0, Y[:, i0: i0 + sample_size]

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
    return X[k, :].reshape(1, -1)

def test_row():
    X = np.array([[0, 1], [1, 2], [3, 4]])
    Y = _row(X, 0)
    assert _ncols(Y) == 2
    assert _nrows(Y) == 1

def _col(X, k):
    return X[:, k].reshape(-1, 1)

def test_col():
    X = np.array([[0, 1], [1, 2], [3, 4]])
    Y = _col(X, 0)
    assert _ncols(Y) == 1
    assert _nrows(Y) == 3

def _slice(X, k):
    return X[:, :, k]

def test_slice():
    X = np.array([[[0], [1]], [[1], [2]], [[3], [4]]])
    Y = _slice(X, 0)
    assert _ncols(Y) == 2
    assert _nrows(Y) == 3

def _set_row(X, k, v):
    X[k, :] = v

def _set_col(X, k, v):
    X[:, k] = v.ravel()

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
    return X[:, :, -(X.shape[-1] - 1):]

def _head_cols(X):
    return X[:, :X.shape[-1] - 1]

def _tail_cols(X):
    return X[:, -(X.shape[-1] - 1):]

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
        if init_X0: self.X0 = _create_noised_values(self.lat_dim, 1)
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
    
    # Evaluation

    def _penalize_low_std_to_mean_ratio(self, X0, P0):
        """
         We expect that the variance must be
         low compared with the mean:
           Mean      Std       penalty
            1         0            undefined
            1         1            100 = 100 * std/mean
            10        1            10 = 100 * std/mean
            20        1            5 = 100 * std/mean
            50        1            2 = 100 * std/mean
        """
        mean = np.abs(np.mean(X0))
        std = np.mean(P0)
        if np.abs(mean) < 1e-3: return 0
        return 100 * std/mean
        
    def _penalize_low_variance(self, X):
        """
        penalty RULE:
        0.0001          10
        0.001           1
        0.01            0.1
        0.1             0.01
        1               0.001
        10              0.0001
        """
        #return 10 ** (-np.log10(np.sum(X) + 1e-100) - 2.0)
        return 0.1/ ((X / (X.mean(axis=1) ** 2)).max())
        #return 0.001/np.sum(X)
    
    def _penalize_inestable_system(self, X):
        """
        penalty
        eigenvalues of X    penalty
        1   1   1           ~ 27
        0.1 0.1 0.1         ~ 0.08
        0.5 0.9 0.5         ~ 8
        """
        return np.sum(np.abs(np.linalg.eig(X)[0])) ** 3

    def _penalize_mean_squared_error(self, Y, ks):
        ##return _mean_squared_error(Y, ks.Ys())
        mse = lambda a, b: np.mean((a.ravel() - b.ravel()) ** 2)
        Ypred = ks.Ys()
        return np.max([mse(Y[k, :], Ypred[k, :]) for k in range(_nrows(Y))])
    
    def _penalize_roughness(self, X):
        return _measure_roughness(X)

    def performance_parameters_line(self, Y):
        (loglikelihood,
        low_std_to_mean_penalty,
        low_variance_Q_penalty,
        low_variance_R_penalty,
        low_variance_P0_penalty,
        system_inestability_penalty,
        mean_squared_error_penalty,
        roughness_X_penalty,
        roughness_Y_penalty) = self.performance_parameters(Y)
        return "LL: {0:.2g} | std/avg: {1:.2g} | lQ: {2:.2g} lR: {3:.2g} lP0: {4:.2g} | inest: {5:.2g} mse: {6:.2g} | roX: {7:.2g} roY: {8:.2g}".format(
                loglikelihood,
                low_std_to_mean_penalty,
                low_variance_Q_penalty,
                low_variance_R_penalty,
                low_variance_P0_penalty,
                system_inestability_penalty,
                mean_squared_error_penalty,
                roughness_X_penalty,
                roughness_Y_penalty)
    
    def performance_parameters(self, Y):
        ks = kalman_smoother_from_parameters(Y, self)
        loglikelihood = ks.loglikelihood()
        low_std_to_mean_penalty = self._penalize_low_std_to_mean_ratio(ks.X0(), ks.P0())
        low_variance_Q_penalty = self._penalize_low_variance(ks.Q())
        low_variance_R_penalty = self._penalize_low_variance(ks.R())
        low_variance_P0_penalty = self._penalize_low_variance(ks.P0())
        system_inestability_penalty = self._penalize_inestable_system(ks.F())
        mean_squared_error_penalty = self._penalize_mean_squared_error(Y, ks)
        roughness_X_penalty = self._penalize_roughness(ks.Xs())
        roughness_Y_penalty = self._penalize_roughness(Y)
        return [
            loglikelihood,
            low_std_to_mean_penalty,
            low_variance_Q_penalty,
            low_variance_R_penalty,
            low_variance_P0_penalty,
            system_inestability_penalty,
            mean_squared_error_penalty,
            roughness_X_penalty,
            roughness_Y_penalty,
        ]


def _create_params_ones_kx1(M, K=[100.0]):
    K = np.array([K]).T
    params = SSMParameters()
    params.F = np.array([[1-1e-10]])
    params.H = K
    params.Q = np.array([[0.001]])
    params.R = 0.001 * _create_noised_diag(_nrows(K), _nrows(K))
    params.X0 = np.array([M])
    params.P0 = np.array([[0.001]])
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
    #print("H:", H)
    #print("X:", Xpred)
    for t in range(_ncols(Xpred)):
        _set_col(Ypred, t, _dot(H, _col(Xpred, t)))
        #print("Ypred", t, ":", _col(Ypred, t))
    return Ypred

def _mean_squared_error(Y, Ypred):
    return np.sum((Y - Ypred) ** 2) / (_ncols(Y) - _nrows(Y) - 1)

def test_mean_squared_error():
    Y = np.array([
        [ 1,  3,  5,  7,  9],
        [11, 13, 15, 17, 19]
    ])
    Ypred = np.array([
        [ 1,  3,  5,  7,  9],
        [11, 13, 15, 17, 19]
    ])
    assert _mean_squared_error(Y, Ypred) == 0
    Ypred = np.array([
        [ 0,  3,  5,  7,  9],
        [11, 13, 14, 17, 17]
    ])
    assert _mean_squared_error(Y, Ypred) == 3.0

def _measure_roughness_original(X):
    # https://stats.stackexchange.com/questions/24607/how-to-measure-smoothness-of-a-time-series-in-r
    diffX = np.diff(X, axis=1)
    return np.round(np.mean(np.std(diffX, axis=1) / (np.abs(np.mean(diffX, axis=1)) + 1e-3)), 2)

def _measure_roughness_proposed(y, M=10):
    cols = y.shape[0]//M
    y = y[: cols * M].reshape(cols, M)
    ystd = (y - y.mean(axis=1).reshape(-1, 1)) / y.std(axis=1).reshape(-1, 1)
    ystd = np.nan_to_num(ystd) #only nan if 0/0, for us = 0
    ystd[np.isinf(ystd)] = 0 #only nan if x/0, for us = 0
    ystd[np.isinf(-ystd)] = 0 #only nan if x/0, for us = 0
    ystd = np.diff(ystd, axis=1).ravel()
    #plt.plot(ystd.ravel()); plt.show();
    return np.mean(np.abs(ystd))

def _measure_roughness(X, M=10):
    return np.mean([_measure_roughness_proposed(X[k, :].ravel(), M) for k in range(X.shape[0])])

def test_measure_roughness():
    X1 = np.array([
        [57, -21, 71, 45, -9, 52, -90, -13, 3, 99, 
         -52, -63, -64, -56, -35, -32, 83, 67, -65, 38, 
         -55, -1, -40, -93, -93, 57, 53, -64, -24, 32, 
         12, 83, -75, -48, 39, 87, 28, -17, 71, 78, 
         72, -57, 64, 80, -60, 67, -89, 14, 62, 60, 
         -4, -19, 18, -4, 10, 51, -51, 74, 2, 15, 
         -41, 71, -56, 99, -30, -95, -67, -44, 65, -46, 
         -21, -70, 95, 72, 41, 11, -98, -72, -70, 75, 
         -28, -2, -79, -21, -3, 86, 0, -58, 79, 14, -96, 
         -63, 84, -100, -55, 85, -94, 53, -49, 27],
        [12, 50, 66, 79, 52, 81, -62, 79, 64, -34, 
         -26, -88, 69, 58, -15, -19, -89, 23, -27, -82, 
         -16, 16, -46, 99, -48, 31, 61, 46, 2, -66, 
         -41, -32, -1, 31, 52, -56, -14, -26, 48, 30, 
         63, 28, 61, 56, -27, 32, -52, -86, 74, -55, 
         67, 97, -30, 24, -42, -67, -99, -40, 49, -19,
         -61, 55, 79, -47, -17, -52, 88, 78, -65, 90, 
         -95, 50, -96, -21, 73, -22, -5, -45, 55, 86, 
         -39, 79, -25, -79, 64, 38, 18, -8, 49, -48, 
         -93, -67, -88, -4, 15, -90, -67, 74, 68, -1]
    ])
    X = X1
    assert np.abs(_measure_roughness(X) - 1.25) < 0.1
    #assert _measure_roughness(X) == 0.0
    X = np.ones(X.shape)
    assert np.abs(_measure_roughness(X) - 0) < 0.1
    #assert _measure_roughness(X) == 2.0
    X = 10 * X1
    assert np.abs(_measure_roughness(X) - 1.25) < 0.1
    X = 1 + 0.0002 * X1
    X[0, :] = 10
    assert np.abs(_measure_roughness(X) - 0.6) < 0.1
    #assert _measure_roughness(X) == 1500.0



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
            #_ncols(self.R()) == _nrows(self.Q()) and
            _ncols(self.R()) == _nrows(self.H())
        ):
            raise ValueError("There is no concordance in the dimension of observed signal. Values:",
                _nrows(self.Y()) == _nrows(self.H()),
                #_ncols(self.R()) == _nrows(self.Q()),
                _ncols(self.R()) == _nrows(self.H()),
                ", ".join("{0}: {1}".format(n, getattr(self, n)().shape) for n in ("Y", "F", "H", "R", "Q", "P0", "Q"))
            )

        if not(
            _nrows(self.P0()) == _ncols(self.P0()) and
            _nrows(self.X0()) == _ncols(self.P0()) and
            _nrows(self.X0()) == _ncols(self.H()) and
            _nrows(self.F()) == _ncols(self.H()) and
            _nrows(self.F()) == _ncols(self.Q()) and
            _ncols(self.F()) == _nrows(self.F()) and
            _ncols(self.Q()) == _nrows(self.Q())
        ):
            raise ValueError("There is no concordance in the dimension of latent signal. Values:",
                _nrows(self.P0()) == _ncols(self.P0()),
                _nrows(self.X0()) == _ncols(self.P0()),
                _nrows(self.X0()) == _ncols(self.H()),
                _nrows(self.F()) == _ncols(self.H()),
                _nrows(self.F()) == _ncols(self.Q()),
                _ncols(self.F()) == _nrows(self.F()),
                _ncols(self.Q()) == _nrows(self.Q()),
                ", ".join("{0}: {1}".format(n, getattr(self, n)().shape) for n in ("Y", "F", "H", "R", "Q", "P0", "Q"))
            )

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
    type_of_likelihood = "smooth"
    
    def __init__(self):
        super().__init__()
        self.smoothed_estimates = SSMEstimated()
    
    def Xs(self): return self.smoothed_estimates.X
    
    def Ps(self): return self.smoothed_estimates.P

    def Ys(self): return self.smoothed_estimates.Y

    def Cs(self): return self.smoothed_estimates.ACV1

    def loglikelihood_smooth(self):
        log_likelihood = 0
        for k in range(1, self.T()):
            Sigma_k = _dot(self.H(), _slice(self.Ps(), k-1), _t(self.H())) + self.R()
            current_likelihood = _mvn_logprobability(_col(self.Y(), k), _dot(self.H(), _col(self.Xs(), k)), Sigma_k)
            if np.isfinite(current_likelihood):
                log_likelihood += current_likelihood
        return log_likelihood / (self.T() - 1)
    
    def loglikelihood_filter(self):
        #http://support.sas.com/documentation/cdl/en/imlug/65547/HTML/default/viewer.htm#imlug_timeseriesexpls_sect035.htm
        #Why this is better:
        #https://stats.stackexchange.com/questions/296598/why-is-the-likelihood-in-kalman-filter-computed-using-filter-results-instead-of
        log_likelihood = 0
        for k in range(1, self.T()):
            Sigma_k = _dot(self.H(), _slice(self.Pf(), k-1), _t(self.H())) + self.R()
            current_likelihood = _mvn_logprobability(_col(self.Y(), k), _dot(self.H(), _col(self.Xf(), k)), Sigma_k)
            if np.isfinite(current_likelihood):
                log_likelihood += current_likelihood
        return log_likelihood / (self.T() - 1)
    
    def loglikelihood_qfunction(self):
        log_likelihood = 0#_mvn_logprobability(self.X0(), self.X0(), self.P0()) <= 0
        for k in range(1, self.T()):
            current_likelihood = _mvn_logprobability(_col(self.Xs(), k), _dot(self.F(), _col(self.Xs(), k - 1)), self.Q())
            if np.isfinite(current_likelihood): #Temporal patch
                log_likelihood += current_likelihood
        for k in range(0, self.T()):
            current_likelihood = _mvn_logprobability(_col(self.Y(), k), _dot(self.H(), _col(self.Xs(), k)), self.R())
            if np.isfinite(current_likelihood):
                log_likelihood += current_likelihood
        return log_likelihood / (self.T() - 1)
    
    def loglikelihood(self):
        if self.type_of_likelihood == "filter":
            return self.loglikelihood_filter()
        if self.type_of_likelihood == "smooth":
            return self.loglikelihood_filter()
        if self.type_of_likelihood == "function-q":
            return self.loglikelihood_filter()
        raise ValueError("Wrong loglikelihood type!")
    
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
            _set_slice(self.Ps(), k, _slice(self.Pf(), k) - _dot(A, _slice(self.Ps(), k + 1) - _slice(self.Pf(), k + 1), _t(A))) #Ghahramani
            _set_col(self.Xs(), k, _col(self.Xf(), k) + _dot(A, _col(self.Xs(), k + 1) - _col(self.Xp(), k + 1)))
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

# it sends a reference of params, not a copy
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

# https://emtiyaz.github.io/papers/TBME-00664-2005.R2-preprint.pdf

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
                _set_slice(ACF, i, _slice(ks.Cs(), i) + _dot(_col(ks.Xs(), i + 1), _t(_col(ks.Xs(), i))))
        
        if self.estimate_H:
            self.parameters.H = _inv(_sum_slices(P))
            H1 = _zero_matrix(O, L)
            for t in range(T):
                H1 += _dot(_col(ks.Y(), t), _t(_col(ks.Xs(), t)))
                #print(" ", t, _dot(_col(ks.Y(), t), _t(_col(ks.Xs(), t))))
            self.parameters.H = _dot(H1, self.parameters.H)
            #print("**",self.parameters.H)
        if self.estimate_R:
            self.parameters.R = _zero_matrix(O, O)
            for t in range(T):
                self.parameters.R += _dot(_col(ks.Y(), t), _t(_col(ks.Y(), t))) - _dot(self.parameters.H, _col(ks.Xs(), t), _t(_col(ks.Y(), t)))
            self.parameters.R /= T
            # Fix math rounding errors
            _set_diag_values_positive(self.parameters.R)
        if self.estimate_F:
            self.parameters.F = _dot(_sum_slices(ACF), _inv(_sum_slices(_head_slices(P))))
        if self.estimate_Q:
            self.parameters.Q = _sum_slices(_tail_slices(P)) - _dot(self.parameters.F, _t(_sum_slices(ACF)))
            self.parameters.Q /= (T - 1)
            _set_diag_values_positive(self.parameters.Q)
        if self.estimate_X0:
            self.parameters.X0 = _col(ks.Xs(), 0)
        if self.estimate_P0:
            self.parameters.P0 = _slice(ks.Ps(), 0)
            _set_diag_values_positive(self.parameters.P0)
        #self.parameters.show(); print("-"*80)

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
    #
    estimator.min_iterations = min_iterations
    estimator.max_iterations = max_iterations
    estimator.min_improvement = min_improvement
    #
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
    #print(records)
    #neoparams.show()
    #params.show()
    #params_orig.show()
    assert (np.abs(np.mean(neoparams.X0) - -50) <= 0.15 * 50), "Failed simulation: mean(X0 pred) != true mean"
    assert (np.abs(np.mean(neoparams.F) - 1) <= 0.15 * 1), "Failed simulation: mean(F pred) != true mean"
    assert (np.abs(np.mean(neoparams.H) - 10) <= 0.15 * 10), "Failed simulation: mean(H pred) != true mean"
    
#####################################################################
# PSO Heuristic SSM Estimator
#####################################################################

resourceParams["trace.intermediates"] = True

class PSOHeuristicEstimatorParticle:
    def __init__(self):
        self.params = SSMParameters()
        self.metric = -1e100
        self.loglikelihood = -1e100
        self.best_params = SSMParameters()
        self.best_metric = -1e150
        self.best_loglikelihood = -1e150
        
        self.penalty_factor_low_variance_Q = 0
        self.penalty_factor_low_variance_R = 0
        self.penalty_factor_low_variance_P0 = 0
        self.penalty_factor_low_std_mean_ratio = 0
        self.penalty_factor_inestable_system = 0
        self.penalty_factor_mse = 0
        self.penalty_factor_roughness_X = 0
        self.penalty_factor_roughness_Y = 0

        self.estimate_F = True
        self.estimate_H = True
        self.estimate_Q = True
        self.estimate_R = True
        self.estimate_X0 = True
        self.estimate_P0 = True
        
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
    
    def set_penalty_factors(self, low_variance_Q=1, low_variance_R=1, low_variance_P0=1, low_std_mean_ratio=1, inestable_system=1, mse=0.5, roughness_X=1, roughness_Y=1):
        self.penalty_factor_low_variance_Q = low_variance_Q
        self.penalty_factor_low_variance_R = low_variance_R
        self.penalty_factor_low_variance_P0 = low_variance_P0
        self.penalty_factor_low_std_mean_ratio = low_std_mean_ratio
        self.penalty_factor_inestable_system = inestable_system
        self.penalty_factor_mse = mse
        self.penalty_factor_roughness_X = roughness_X
        self.penalty_factor_roughness_Y = roughness_Y
    '''
    def _penalize_low_std_to_mean_ratio(self, ks):
        """
         We expect that the variance must be
         low compared with the mean:
           Mean      Std       penalty
            1         0            undefined
            1         1            100 = 100 * std/mean
            10        1            10 = 100 * std/mean
            20        1            5 = 100 * std/mean
            50        1            2 = 100 * std/mean
        """
        mean = np.abs(np.mean(ks.X0()))
        std = np.mean(ks.P0())
        if np.abs(mean) < 1e-3: return 0
        return 100 * std/mean
        
    def _penalize_low_variance(self, X):
        """
        penalty RULE:
        0.0001          10
        0.001           1
        0.01            0.1
        0.1             0.01
        1               0.001
        10              0.0001
        """
        #return 10 ** (-np.log10(np.sum(X) + 1e-100) - 2.0)
        return 0.1/ ((X / (X.mean(axis=1) ** 2)).max())
        #return 0.001/np.sum(X)
    
    def _penalize_inestable_system(self, X):
        """
        penalty
        eigenvalues of X    penalty
        1   1   1           ~ 27
        0.1 0.1 0.1         ~ 0.08
        0.5 0.9 0.5         ~ 8
        """
        return np.sum(np.abs(np.linalg.eig(X)[0])) ** 3

    def _penalize_mean_squared_error(self, Y, ks):
        return _mean_squared_error(Y, ks.Ys())
    
    def _penalize_roughness(self, X):
        return _measure_roughness(X)

    def evaluate(self, Y, index=0):
        ks = kalman_smoother_from_parameters(Y, self.params)
        self.loglikelihood = ks.loglikelihood()
        self.metric = self.loglikelihood
        self.metric -= self.penalty_factor_low_std_mean_ratio * self._penalize_low_std_to_mean_ratio(ks)
        self.metric -= self.penalty_factor_low_variance_Q * self._penalize_low_variance(ks.Q())
        self.metric -= self.penalty_factor_low_variance_R * self._penalize_low_variance(ks.R())
        self.metric -= self.penalty_factor_low_variance_P0 * self._penalize_low_variance(ks.P0())
        self.metric -= self.penalty_factor_inestable_system * self._penalize_inestable_system(ks.F())
        self.metric -= self.penalty_factor_mse * self._penalize_mean_squared_error(Y, ks)
        self.metric -= self.penalty_factor_roughness_X * self._penalize_roughness(ks.Xs())
        self.metric -= self.penalty_factor_roughness_Y * self._penalize_roughness(Y)
        if resourceParams["trace.intermediates"]:
            print("*****",
                self.metric,
                "====",
                np.round(self.loglikelihood, 2),
                "|",
                np.round(self._penalize_low_std_to_mean_ratio(ks), 2),
                "|",
                np.round(self._penalize_low_variance(ks.Q()), 2),
                np.round(self._penalize_low_variance(ks.R()), 2),
                np.round(self._penalize_low_variance(ks.P0()), 2),
                "|",
                np.round(self._penalize_inestable_system(ks.F()), 2),
                np.round(self._penalize_mean_squared_error(Y, ks), 2),
                "|",
                np.round(self._penalize_roughness(ks.Xs()), 2),
                np.round(self._penalize_roughness(Y), 2),
            )
        if self.metric > self.best_metric:
            self.best_metric = self.metric
            self.best_params = self.params.copy()
            self.best_loglikelihood = self.loglikelihood
    '''
    
    def evaluate(self, Y, index=0):
        (loglikelihood,
        low_std_to_mean_penalty,
        low_variance_Q_penalty,
        low_variance_R_penalty,
        low_variance_P0_penalty,
        system_inestability_penalty,
        mean_squared_error_penalty,
        roughness_X_penalty,
        roughness_Y_penalty) = self.params.performance_parameters(Y)
        self.loglikelihood = loglikelihood
        self.metric = loglikelihood
        self.metric -= self.penalty_factor_low_std_mean_ratio * low_std_to_mean_penalty
        self.metric -= self.penalty_factor_low_variance_Q * low_variance_Q_penalty
        self.metric -= self.penalty_factor_low_variance_R * low_variance_R_penalty
        self.metric -= self.penalty_factor_low_variance_P0 * low_variance_P0_penalty
        self.metric -= self.penalty_factor_inestable_system * system_inestability_penalty
        self.metric -= self.penalty_factor_mse * mean_squared_error_penalty
        self.metric -= self.penalty_factor_roughness_X * roughness_X_penalty
        self.metric -= self.penalty_factor_roughness_Y * roughness_Y_penalty
        if resourceParams["trace.intermediates"]:
            print("*****",
                self.metric,
                "====",
                self.params.performance_parameters_line(Y)
            )
        if self.metric > self.best_metric:
            self.best_metric = self.metric
            self.best_params = self.params.copy()
            self.best_loglikelihood = self.loglikelihood
    
    def set_movable_params(self, est_F, est_H, est_Q, est_R, est_X0, est_P0):
        self.estimate_F = est_F
        self.estimate_H = est_H
        self.estimate_Q = est_Q
        self.estimate_R = est_R
        self.estimate_X0 = est_X0
        self.estimate_P0 = est_P0
        #!###print("====>", self.estimate_F, self.estimate_H, self.estimate_Q, self.estimate_R, self.estimate_X0, self.estimate_P0,)

    #def __move_fix(self, best_particle):
    def move(self, best_particle):
        #print("==**>", self.estimate_F, self.estimate_H, self.estimate_Q, self.estimate_R, self.estimate_X0, self.estimate_P0,)
        move_to_self_best = 2 * np.random.uniform()
        move_to_global_best = 2 * np.random.uniform()
        if self.estimate_F:
            self.params.F += move_to_self_best * (self.best_params.F - self.params.F) + move_to_global_best * (best_particle.best_params.F - self.params.F)
        if self.estimate_H:
            self.params.H += move_to_self_best * (self.best_params.H - self.params.H) + move_to_global_best * (best_particle.best_params.H - self.params.H)
        if self.estimate_Q:
            self.params.Q += move_to_self_best * (self.best_params.Q - self.params.Q) + move_to_global_best * (best_particle.best_params.Q - self.params.Q)
            self.params.Q = 0.5 * (self.params.Q + _t(self.params.Q))
            _set_diag_values_positive(self.params.Q)
        if self.estimate_R:
            self.params.R += move_to_self_best * (self.best_params.R - self.params.R) + move_to_global_best * (best_particle.best_params.R - self.params.R)
            self.params.R = 0.5 * (self.params.R + _t(self.params.R))
            _set_diag_values_positive(self.params.R)
        if self.estimate_X0:
            self.params.X0 += move_to_self_best * (self.best_params.X0 - self.params.X0) + move_to_global_best * (best_particle.best_params.X0 - self.params.X0)
        if self.estimate_P0:
            self.params.P0 += move_to_self_best * (self.best_params.P0 - self.params.P0) + move_to_global_best * (best_particle.best_params.P0 - self.params.P0)
            self.params.P0 = 0.5 * (self.params.P0 + _t(self.params.P0))
            _set_diag_values_positive(self.params.P0)
    
    def move_flexible(self, best_particle):
        #print("==**>", self.estimate_F, self.estimate_H, self.estimate_Q, self.estimate_R, self.estimate_X0, self.estimate_P0,)
        move_to_self_best = lambda: 2 * np.random.uniform()
        move_to_global_best = lambda: 2 * np.random.uniform()
        if self.estimate_F:
            self.params.F += move_to_self_best() * (self.best_params.F - self.params.F) + move_to_global_best() * (best_particle.best_params.F - self.params.F)
        if self.estimate_H:
            self.params.H += move_to_self_best() * (self.best_params.H - self.params.H) + move_to_global_best() * (best_particle.best_params.H - self.params.H)
        if self.estimate_Q:
            self.params.Q += move_to_self_best() * (self.best_params.Q - self.params.Q) + move_to_global_best() * (best_particle.best_params.Q - self.params.Q)
            self.params.Q = 0.5 * (self.params.Q + _t(self.params.Q))
            _set_diag_values_positive(self.params.Q)
        if self.estimate_R:
            self.params.R += move_to_self_best() * (self.best_params.R - self.params.R) + move_to_global_best() * (best_particle.best_params.R - self.params.R)
            self.params.R = 0.5 * (self.params.R + _t(self.params.R))
            _set_diag_values_positive(self.params.R)
        if self.estimate_X0:
            self.params.X0 += move_to_self_best() * (self.best_params.X0 - self.params.X0) + move_to_global_best() * (best_particle.best_params.X0 - self.params.X0)
        if self.estimate_P0:
            self.params.P0 += move_to_self_best() * (self.best_params.P0 - self.params.P0) + move_to_global_best() * (best_particle.best_params.P0 - self.params.P0)
            self.params.P0 = 0.5 * (self.params.P0 + _t(self.params.P0))
            _set_diag_values_positive(self.params.P0)
    
    def copy_best_from(self, other, force_copy=False):
        if other.best_metric > self.metric or force_copy:
            self.metric = other.best_metric
            self.best_metric = other.best_metric
            self.loglikelihood = other.loglikelihood
            self.best_loglikelihood = other.best_loglikelihood
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
        self.population_size = 50
        self.particles = []
        self.best_particle = None
        
        self.penalty_factor_low_variance_Q = 0.5
        self.penalty_factor_low_variance_R = 0.5
        self.penalty_factor_low_variance_P0 = 0.5
        self.penalty_factor_low_std_mean_ratio = 0.5
        self.penalty_factor_inestable_system = 1
        self.penalty_factor_mse = 0.25
        self.penalty_factor_roughness_X = 2
        self.penalty_factor_roughness_Y = 2

    def _create_particle(self):
        return PSOHeuristicEstimatorParticle()

    def set_parameters(self, Y, parameters=None, est_F=True, est_H=True, est_Q=True, est_R=True, est_X0=True, est_P0=True, lat_dim=None):
        #!###print("****==>", self.estimate_F, self.estimate_H, self.estimate_Q, self.estimate_R, self.estimate_X0, self.estimate_P0,)
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
        #self.sample_size = _ncols(Y)
        self.estimate_F = est_F
        self.estimate_H = est_H
        self.estimate_Q = est_Q
        self.estimate_R = est_R
        self.estimate_X0 = est_X0
        self.estimate_P0 = est_P0
        #!###print("****==>", self.estimate_F, self.estimate_H, self.estimate_Q, self.estimate_R, self.estimate_X0, self.estimate_P0,)
        #
        self.best_particle = self._create_particle()
        self.particles = []
        #parameters.show()
        for i in range(self.population_size):
            self.particles.append(self._create_particle())
            if i == 0:
                self.particles[i].init_with_parameters(_nrows(Y), parameters.copy(), False, False, False, False, False, False, parameters.lat_dim)
            #if i < min(3, int(self.population_size * 0.5)):
            #    self.particles[i].init_with_parameters(_nrows(Y), parameters.copy(), False, False, False, False, False, False, parameters.lat_dim)
            #elif min(3, int(self.population_size * 0.5)) <= i and i < min(5, int(self.population_size * 0.6)):
            #    self.particles[i].init_with_parameters(_nrows(Y), parameters.copy(), True, True, True, True, True, True, parameters.lat_dim)
            else:
                self.particles[i].init_with_parameters(_nrows(Y), parameters.copy(), est_F, est_H, est_Q, est_R, est_X0, est_P0, lat_dim)
            #self.particles[i].params.show()
            self.particles[i].set_movable_params(est_F, est_H, est_Q, est_R, est_X0, est_P0)
            #!###s = self.particles[i]
            #!###print("****==>", s.estimate_F, s.estimate_H, s.estimate_Q, s.estimate_R, s.estimate_X0, s.estimate_P0,)

            self.particles[i].set_penalty_factors(
                self.penalty_factor_low_variance_Q,
                self.penalty_factor_low_variance_R,
                self.penalty_factor_low_variance_P0,
                self.penalty_factor_low_std_mean_ratio,
                self.penalty_factor_inestable_system,
                self.penalty_factor_mse,
                self.penalty_factor_roughness_X,
                self.penalty_factor_roughness_Y,
            )
            y0, subY = _subsample(self.Y, self.sample_size)
            self.particles[i].evaluate(subY, y0)
            self.best_particle.copy_best_from(self.particles[i], True)
            ###print(" **  ", self.particles[i].metric)
            ###self.particles[i].params.show()
            #print("."*80); self.particles[i].params.show()
        #
        ###
        #self.parameters.show()
        self.parameters.copy_from(self.best_particle.best_params)
        ###
        #self.parameters.show()
        #

    def estimation_iteration_heuristic(self):
        for i in range(self.population_size):
            #self.particles[i].evaluate(self.Y)
            #print("-"*80); self.particles[i].params.show()
            y0, subY = _subsample(self.Y, self.sample_size)
            try:
                self.particles[i].evaluate(subY, y0)
            except np.linalg.LinAlgError: #Avoid SVD convergence issues
                self.particles[i].metric = -1e100
                self.particles[i].loglikelihood = -1e100
            self.best_particle.copy_best_from(self.particles[i])
            self.particles[i].move(self.best_particle)
            #print("."*80); self.particles[i].params.show()
        self.loglikelihood_record.append(self.best_particle.best_loglikelihood)
        self.parameters.copy_from(self.best_particle.best_params)
        if resourceParams["trace.intermediates"]:
            ks = kalman_smoother_from_parameters(self.Y, self.best_particle.best_params)
            print(" >>>> ", self.best_particle.metric,
                "====",
                self.best_particle.params.performance_parameters_line(self.Y)
            )
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


def _test_pure_pso_1():
    params_orig = _create_params_ones_kx1([-50], [10])
    params = _create_params_ones_kx1([-50], [10])
    params.X0[0, 0] += 1
    params.H[0, 0] -= 0.3
    params.F[0, 0] -= 0.1
    #params.show()
    x, y = params.simulate(100)
    kf = PurePSOHeuristicEstimator()
    #kf.penalty_factor_roughness_X = 1
    #kf.penalty_factor_roughness_Y = 1
    kf.set_parameters(y, params)
    #kf.estimate_parameters()
    kf.parameters.show()
    if resourceParams["trace.intermediates"]:
        print(kf.loglikelihood_record)
    s = kf.smoother()
    s.smooth()
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

def _test_pure_pso_2():
    params_orig = _create_params_ones_kx1([-50], [10])
    params = _create_params_ones_kx1([-50], [10])
    #params.X0[0, 0] += 1
    #params.H[0, 0] -= 0.3
    #params.F[0, 0] -= 0.1
    #params.show()
    x, y = params.simulate(100)
    kf = PurePSOHeuristicEstimator()
    #kf.penalty_factor_roughness_X = 1
    #kf.penalty_factor_roughness_Y = 1
    kf.set_parameters(y, params)
    kf.estimate_parameters()
    #kf.parameters.show()
    #params.show()
    #params_orig.show()
    s = kf.smoother()
    s.smooth()
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
def estimate_using_pso(Y, estimates="", F0=None, H0=None, Q0=None, R0=None, X00=None, P00=None, min_iterations=5, max_iterations=20, min_improvement=0.01, lat_dim=None, sample_size=30, population_size=50, penalty_factors=(0.5, 0.5, 0.5, 0.5, 1, 0.25, 2.0, 2.0)):
    estimator = PurePSOHeuristicEstimator()
    estimator.Y = Y
    estimator.estimate_F = "F" in estimates
    estimator.estimate_H = "H" in estimates
    estimator.estimate_Q = "Q" in estimates
    estimator.estimate_R = "R" in estimates
    estimator.estimate_X0 = "X0" in estimates
    estimator.estimate_P0 = "P0" in estimates
    
    estimator.sample_size = sample_size
    estimator.population_size = population_size
    estimator.min_iterations = min_iterations
    estimator.max_iterations = max_iterations
    estimator.min_improvement = min_improvement
    
    if isinstance(penalty_factors, dict):
        penalty_factors = [
            penalty_factors.get("low_variance_Q", 0.5),
            penalty_factors.get("low_variance_R", 0.5),
            penalty_factors.get("low_variance_P0", 0.5),
            penalty_factors.get("low_std_mean_ratio", 0.5),
            penalty_factors.get("inestable_system",1),
            penalty_factors.get("mse", 0.25),
            penalty_factors.get("roughness_X", 0.5),
            penalty_factors.get("roughness_Y", 0.5),
        ]
    estimator.penalty_factor_low_variance_Q = penalty_factors[0]
    estimator.penalty_factor_low_variance_R = penalty_factors[1]
    estimator.penalty_factor_low_variance_P0 = penalty_factors[2]
    estimator.penalty_factor_low_std_mean_ratio = penalty_factors[3]
    estimator.penalty_factor_inestable_system = penalty_factors[4]
    estimator.penalty_factor_mse = penalty_factors[5]
    estimator.penalty_factor_roughness_X = penalty_factors[6]
    estimator.penalty_factor_roughness_Y = penalty_factors[7]
    
    parameters = SSMParameters()
    parameters.F = F0
    if F0 is not None:
        parameters.lat_dim = _nrows(F0)
    parameters.H = H0
    if H0 is not None:
        parameters.lat_dim = _ncols(H0)
    parameters.Q = Q0
    if Q0 is not None:
        parameters.lat_dim = _ncols(Q0)
    parameters.R = R0
    parameters.X0 = X00
    if X00 is not None:
        parameters.lat_dim = _nrows(X00)
    parameters.P0 = P00
    if P00 is not None:
        parameters.lat_dim = _ncols(P00)
    if lat_dim is not None:
        parameters.lat_dim = lat_dim
    parameters.obs_dim = _nrows(Y)
    parameters.obs_dim = _nrows(Y)
    parameters.random_initialize(F0 is None, H0 is None, Q0 is None, R0 is None, X00 is None, P00 is None)
    estimator.set_parameters(Y, parameters, "F" in estimates, "H" in estimates, "Q" in estimates, "R" in estimates, "X0" in estimates, "P0" in estimates, lat_dim=None)
    estimator.estimate_parameters()
    ks = estimator.smoother()
    ks.smooth()
    return ks, estimator.loglikelihood_record

def _test_pure_pso_3():
    params = _create_params_ones_kx1([-50], [10])
    x, y = params.simulate(1000)
    ks, records = estimate_using_pso(y,
        estimates="F H Q R X0 P0",
        F0=params.F, H0=params.H, Q0=params.Q, R0=params.R, X00=params.X0, P00=params.P0,
        min_iterations=5, max_iterations=30, min_improvement=0.01, lat_dim=None,
        sample_size=30, population_size=50,
        #penalty_factors=(0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 0.5, 0.5)
    )
    neoparams = ks.parameters
    #print(records)
    #neoparams.show()
    #params.show()
    #params_orig.show()
    assert (np.abs(np.mean(neoparams.X0) - -50) <= 0.15 * 50), "Failed simulation: mean(X0 pred) != true mean"
    assert (np.abs(np.mean(neoparams.F) - 1) <= 0.15 * 1), "Failed simulation: mean(F pred) != true mean"
    assert (np.abs(np.mean(neoparams.H) - 10) <= 0.15 * 10), "Failed simulation: mean(H pred) != true mean"

def _test_pure_pso_4():
    params = _create_params_ones_kx1([-50], [10])
    x, y = params.simulate(1000)
    ks, records = estimate_using_pso(y,
        estimates="F H Q R X0 P0",
        F0=params.F, H0=params.H, Q0=params.Q, R0=params.R, X00=params.X0, P00=params.P0,
        min_iterations=10, max_iterations=20, min_improvement=0.01, lat_dim=None,
        sample_size=10, population_size=100,
        #penalty_factors=(0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 0.5, 0.5)
    )
    neoparams = ks.parameters
    #print(records)
    #neoparams.show()
    #params.show()
    #params_orig.show()
    assert (np.abs(np.mean(neoparams.X0) - -50) <= 0.15 * 50), "Failed simulation: mean(X0 pred) != true mean"
    assert (np.abs(np.mean(neoparams.F) - 1) <= 0.15 * 1), "Failed simulation: mean(F pred) != true mean"
    assert (np.abs(np.mean(neoparams.H) - 10) <= 0.15 * 10), "Failed simulation: mean(H pred) != true mean"


#####################################################################
# LSE Heuristic SSM Estimator
#####################################################################
class LSEHeuristicEstimatorParticle(PSOHeuristicEstimatorParticle):
    def __init__(self):
        super().__init__()
    
    def improve_params(self, Y, index=0):
        #self.params.show(); print()
        try:
            ks = kalman_smoother_from_parameters(Y, self.params.copy())
            X = ks.Xs()
            if self.estimate_H:
                # Y = H X + N(0, R)
                # <H> = Y * X' * inv(X * X')' 
                self.params.H = _dot(Y, _t(X), _t(_inv(_dot(X, _t(X)))))
            if self.estimate_R:
                # <R> = var(Y - H X)
                self.params.R = _covariance_matrix_estimation(Y - _dot(self.params.H, X))
            if self.estimate_F:
                # X[1..T] = F X[0..T-1] + N(0, R)
                # <F> = X1 * X0' * inv(X0 * X0')'
                X0 = _head_cols(X)
                X1 = _tail_cols(X)
                self.params.F = _dot(X1, _t(X0), _t(_inv(_dot(X0, _t(X0)))))
            inv_H = _inv(self.params.H)
            if self.estimate_X0:
                # Y[0] = H X[0] + N(0, R)
                # <X[0]> = inv(H) Y[0]
                X0 = _col(X, 0)#_dot(inv_H, _col(Y, 0))
                inv_F = _inv(self.params.F)
                # IMPROVE!
                for _ in range(index):
                    X0 = _dot(inv_F, X0)
                self.params.X0 = X0
            if self.estimate_P0:
                # <X[0]> = inv(H) Y[0] => VAR<X[0]> = inv(H) var(Y[0]) inv(H)' 
                # P[0] = inv(H) R inv(H)'
                self.params.P0 = _dot(inv_H, self.params.R, _t(inv_H))
            if self.estimate_Q:
                # <Q> = var(X1 - F X0)
                X0 = _head_cols(X)
                X1 = _tail_cols(X)
                self.params.Q = _covariance_matrix_estimation(X1 - _dot(self.params.F, X0))
            #self.params.show()
            #sys.exit(0)
        except np.linalg.LinAlgError:
            pass #DO something
    
    def evaluate(self, Y, index=0):
        super().evaluate(Y, index)
        prev_metric = self.metric
        prev_parameters = self.params.copy()
        self.improve_params(Y, index)
        super().evaluate(Y, index)
        if prev_metric > self.metric:
            self.metric = prev_metric
            self.params.copy_from(prev_parameters)

class LSEHeuristicEstimator(PurePSOHeuristicEstimator):
    def __init__(self):
        super().__init__()

    def _create_particle(self):
        return LSEHeuristicEstimatorParticle()

def _test_pure_lse_pso_1():
    params_orig = _create_params_ones_kx1([-50], [10])
    params = _create_params_ones_kx1([-50], [10])
    params.X0[0, 0] += 1
    params.H[0, 0] -= 0.3
    params.F[0, 0] -= 0.1
    #params.show()
    x, y = params.simulate(100)
    #print(y)
    kf = LSEHeuristicEstimator()
    #kf.penalty_factor_roughness_X = 1
    #kf.penalty_factor_roughness_Y = 1
    kf.set_parameters(y, params)
    kf.estimate_parameters()
    kf.parameters.show()
    if resourceParams["trace.intermediates"]:
        print(kf.loglikelihood_record)
    s = kf.smoother()
    s.smooth()
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

def _test_pure_lse_pso_2():
    params_orig = _create_params_ones_kx1([-50], [10])
    params = _create_params_ones_kx1([-50], [10])
    #params.X0[0, 0] += 1
    #params.H[0, 0] -= 0.3
    #params.F[0, 0] -= 0.1
    #params.show()
    x, y = params.simulate(5000)
    kf = LSEHeuristicEstimator()
    #kf.penalty_factor_roughness_X = 1
    #kf.penalty_factor_roughness_Y = 1
    kf.set_parameters(y, params)
    kf.estimate_parameters()
    #kf.parameters.show()
    #params.show()
    #params_orig.show()
    s = kf.smoother()
    s.smooth()
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
def estimate_using_lse_pso(Y, estimates="", F0=None, H0=None, Q0=None, R0=None, X00=None, P00=None, min_iterations=5, max_iterations=20, min_improvement=0.01, lat_dim=None, sample_size=30, population_size=50, penalty_factors=(0.5, 0.5, 0.5, 0.5, 1, 0.25, 2.0, 2.0)):
    estimator = LSEHeuristicEstimator()
    estimator.Y = Y
    estimator.estimate_F = "F" in estimates
    estimator.estimate_H = "H" in estimates
    estimator.estimate_Q = "Q" in estimates
    estimator.estimate_R = "R" in estimates
    estimator.estimate_X0 = "X0" in estimates
    estimator.estimate_P0 = "P0" in estimates
    
    estimator.sample_size = sample_size
    estimator.population_size = population_size
    estimator.min_iterations = min_iterations
    estimator.max_iterations = max_iterations
    estimator.min_improvement = min_improvement

    if isinstance(penalty_factors, dict):
        penalty_factors = [
            penalty_factors.get("low_variance_Q", 0.5),
            penalty_factors.get("low_variance_R", 0.5),
            penalty_factors.get("low_variance_P0", 0.5),
            penalty_factors.get("low_std_mean_ratio", 0.5),
            penalty_factors.get("inestable_system",1),
            penalty_factors.get("mse", 0.25),
            penalty_factors.get("roughness_X", 0.5),
            penalty_factors.get("roughness_Y", 0.5),
        ]
    estimator.penalty_factor_low_variance_Q = penalty_factors[0]
    estimator.penalty_factor_low_variance_R = penalty_factors[1]
    estimator.penalty_factor_low_variance_P0 = penalty_factors[2]
    estimator.penalty_factor_low_std_mean_ratio = penalty_factors[3]
    estimator.penalty_factor_inestable_system = penalty_factors[4]
    estimator.penalty_factor_mse = penalty_factors[5]
    estimator.penalty_factor_roughness_X = penalty_factors[6]
    estimator.penalty_factor_roughness_Y = penalty_factors[7]
    
    parameters = SSMParameters()
    parameters.F = F0
    if F0 is not None:
        parameters.lat_dim = _nrows(F0)
    parameters.H = H0
    if H0 is not None:
        parameters.lat_dim = _ncols(H0)
    parameters.Q = Q0
    if Q0 is not None:
        parameters.lat_dim = _ncols(Q0)
    parameters.R = R0
    parameters.X0 = X00
    if X00 is not None:
        parameters.lat_dim = _nrows(X00)
    parameters.P0 = P00
    if P00 is not None:
        parameters.lat_dim = _ncols(P00)
    if lat_dim is not None:
        parameters.lat_dim = lat_dim
    parameters.obs_dim = _nrows(Y)
    parameters.obs_dim = _nrows(Y)
    parameters.random_initialize(F0 is None, H0 is None, Q0 is None, R0 is None, X00 is None, P00 is None)
    estimator.set_parameters(Y, parameters, "F" in estimates, "H" in estimates, "Q" in estimates, "R" in estimates, "X0" in estimates, "P0" in estimates, lat_dim=None)
    estimator.estimate_parameters()
    ks = estimator.smoother()
    ks.smooth()
    return ks, estimator.loglikelihood_record

def _test_pure_lse_pso_3():
    params = _create_params_ones_kx1([-50], [10])
    x, y = params.simulate(1000)
    ks, records = estimate_using_lse_pso(y,
        estimates="F H Q R X0 P0",
        F0=params.F, H0=params.H, Q0=params.Q, R0=params.R, X00=params.X0, P00=params.P0,
        min_iterations=5, max_iterations=30, min_improvement=0.01, lat_dim=None,
        sample_size=30, population_size=50,
        #penalty_factors=(0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 0.5, 0.5)
    )
    neoparams = ks.parameters
    #print(records)
    #neoparams.show()
    #params.show()
    #params_orig.show()
    assert (np.abs(np.mean(neoparams.X0) - -50) <= 0.15 * 50), "Failed simulation: mean(X0 pred) != true mean"
    assert (np.abs(np.mean(neoparams.F) - 1) <= 0.15 * 1), "Failed simulation: mean(F pred) != true mean"
    assert (np.abs(np.mean(neoparams.H) - 10) <= 0.15 * 10), "Failed simulation: mean(H pred) != true mean"

def _test_pure_lse_pso_4():
    params = _create_params_ones_kx1([-50], [10])
    x, y = params.simulate(1000)
    ks, records = estimate_using_lse_pso(y,
        estimates="F H Q R X0 P0",
        F0=params.F, H0=params.H, Q0=params.Q, R0=params.R, X00=params.X0, P00=params.P0,
        min_iterations=10, max_iterations=20, min_improvement=0.01, lat_dim=None,
        sample_size=10, population_size=100,
        #penalty_factors=(0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 0.5, 0.5)
    )
    neoparams = ks.parameters
    #print(records)
    #neoparams.show()
    #params.show()
    #params_orig.show()
    assert (np.abs(np.mean(neoparams.X0) - -50) <= 0.15 * 50), "Failed simulation: mean(X0 pred) != true mean"
    assert (np.abs(np.mean(neoparams.F) - 1) <= 0.15 * 1), "Failed simulation: mean(F pred) != true mean"
    assert (np.abs(np.mean(neoparams.H) - 10) <= 0.15 * 10), "Failed simulation: mean(H pred) != true mean"



#####################################################################
# EM Heuristic SSM Estimator
#####################################################################
class EMHeuristicEstimatorParticle(PSOHeuristicEstimatorParticle):
    def __init__(self):
        super().__init__()
    
    def improve_params(self, Y, index=0):
        subestimator = ExpectationMaximizationEstimator()
        subestimator.Y = Y
        subestimator.estimate_F = self.estimate_F
        subestimator.estimate_H = self.estimate_H
        subestimator.estimate_Q = self.estimate_Q
        subestimator.estimate_R = self.estimate_R
        subestimator.estimate_X0 = self.estimate_X0
        subestimator.estimate_P0 = self.estimate_P0
        #print(
        #    "subestimator:",
        #    subestimator.estimate_F,
        #    subestimator.estimate_H,
        #    subestimator.estimate_Q,
        #    subestimator.estimate_R,
        #    subestimator.estimate_X0,
        #    subestimator.estimate_P0,
        #)
        subestimator.parameters = self.params
        #estimator.parameters.random_initialize(F0 is None, H0 is None, Q0 is None, R0 is None, X00 is None, P00 is None)
        subestimator.estimation_iteration()
    
    def evaluate(self, Y, index=0):
        super().evaluate(Y, index)
        prev_metric = self.metric
        prev_parameters = self.params.copy()
        self.improve_params(Y, index)
        super().evaluate(Y, index)
        if prev_metric > self.metric:
            self.metric = prev_metric
            self.params.copy_from(prev_parameters)

class EMHeuristicEstimator(PurePSOHeuristicEstimator):
    def __init__(self):
        super().__init__()

    def _create_particle(self):
        return EMHeuristicEstimatorParticle()


def test_pure_em_pso_1():
    params_orig = _create_params_ones_kx1([-50], [10])
    params = _create_params_ones_kx1([-50], [10])
    params.X0[0, 0] += 1
    params.H[0, 0] -= 0.3
    params.F[0, 0] -= 0.1
    #params.show()
    x, y = params.simulate(100)
    kf = EMHeuristicEstimator()
    #kf.penalty_factor_roughness_X = 1
    #kf.penalty_factor_roughness_Y = 1
    kf.set_parameters(y, params)
    kf.estimate_parameters()
    kf.parameters.show()
    if resourceParams["trace.intermediates"]:
        print(kf.loglikelihood_record)
    s = kf.smoother()
    s.smooth()
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

def test_pure_em_pso_2():
    params_orig = _create_params_ones_kx1([-50], [10])
    params = _create_params_ones_kx1([-50], [10])
    #params.X0[0, 0] += 1
    #params.H[0, 0] -= 0.3
    #params.F[0, 0] -= 0.1
    #params.show()
    x, y = params.simulate(100)
    kf = EMHeuristicEstimator()
    #kf.penalty_factor_roughness_X = 1
    #kf.penalty_factor_roughness_Y = 1
    kf.set_parameters(y, params)
    kf.estimate_parameters()
    #kf.parameters.show()
    #params.show()
    #params_orig.show()
    s = kf.smoother()
    s.smooth()
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
def estimate_using_em_pso(Y, estimates="", F0=None, H0=None, Q0=None, R0=None, X00=None, P00=None, min_iterations=5, max_iterations=20, min_improvement=0.01, lat_dim=None, sample_size=30, population_size=50, penalty_factors=(0.5, 0.5, 0.5, 0.5, 1, 0.25, 2.0, 2.0)):
    estimator = EMHeuristicEstimator()
    estimator.Y = Y
    estimator.estimate_F = "F" in estimates
    estimator.estimate_H = "H" in estimates
    estimator.estimate_Q = "Q" in estimates
    estimator.estimate_R = "R" in estimates
    estimator.estimate_X0 = "X0" in estimates
    estimator.estimate_P0 = "P0" in estimates
    
    estimator.sample_size = sample_size
    estimator.population_size = population_size
    estimator.min_iterations = min_iterations
    estimator.max_iterations = max_iterations
    estimator.min_improvement = min_improvement

    if isinstance(penalty_factors, dict):
        penalty_factors = [
            penalty_factors.get("low_variance_Q", 0.5),
            penalty_factors.get("low_variance_R", 0.5),
            penalty_factors.get("low_variance_P0", 0.5),
            penalty_factors.get("low_std_mean_ratio", 0.5),
            penalty_factors.get("inestable_system",1),
            penalty_factors.get("mse", 0.25),
            penalty_factors.get("roughness_X", 0.5),
            penalty_factors.get("roughness_Y", 0.5),
        ]
    estimator.penalty_factor_low_variance_Q = penalty_factors[0]
    estimator.penalty_factor_low_variance_R = penalty_factors[1]
    estimator.penalty_factor_low_variance_P0 = penalty_factors[2]
    estimator.penalty_factor_low_std_mean_ratio = penalty_factors[3]
    estimator.penalty_factor_inestable_system = penalty_factors[4]
    estimator.penalty_factor_mse = penalty_factors[5]
    estimator.penalty_factor_roughness_X = penalty_factors[6]
    estimator.penalty_factor_roughness_Y = penalty_factors[7]
    
    parameters = SSMParameters()
    parameters.F = F0
    if F0 is not None:
        parameters.lat_dim = _nrows(F0)
    parameters.H = H0
    if H0 is not None:
        parameters.lat_dim = _ncols(H0)
    parameters.Q = Q0
    if Q0 is not None:
        parameters.lat_dim = _ncols(Q0)
    parameters.R = R0
    parameters.X0 = X00
    if X00 is not None:
        parameters.lat_dim = _nrows(X00)
    parameters.P0 = P00
    if P00 is not None:
        parameters.lat_dim = _ncols(P00)
    if lat_dim is not None:
        parameters.lat_dim = lat_dim
    parameters.obs_dim = _nrows(Y)
    parameters.obs_dim = _nrows(Y)
    parameters.random_initialize(F0 is None, H0 is None, Q0 is None, R0 is None, X00 is None, P00 is None)
    estimator.set_parameters(Y, parameters, "F" in estimates, "H" in estimates, "Q" in estimates, "R" in estimates, "X0" in estimates, "P0" in estimates, lat_dim=None)
    estimator.estimate_parameters()
    ks = estimator.smoother()
    ks.smooth()
    return ks, estimator.loglikelihood_record

def _test_pure_em_pso_3():
    params = _create_params_ones_kx1([-50], [10])
    x, y = params.simulate(1000)
    ks, records = estimate_using_em_pso(y,
        estimates="F H Q R X0 P0",
        F0=params.F, H0=params.H, Q0=params.Q, R0=params.R, X00=params.X0, P00=params.P0,
        min_iterations=5, max_iterations=30, min_improvement=0.01, lat_dim=None,
        sample_size=30, population_size=50,
        #penalty_factors=(0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 0.5, 0.5)
    )
    neoparams = ks.parameters
    #print(records)
    #neoparams.show()
    #params.show()
    #params_orig.show()
    assert (np.abs(np.mean(neoparams.X0) - -50) <= 0.15 * 50), "Failed simulation: mean(X0 pred) != true mean"
    assert (np.abs(np.mean(neoparams.F) - 1) <= 0.15 * 1), "Failed simulation: mean(F pred) != true mean"
    assert (np.abs(np.mean(neoparams.H) - 10) <= 0.15 * 10), "Failed simulation: mean(H pred) != true mean"

def _test_pure_em_pso_4():
    params = _create_params_ones_kx1([-50], [10])
    x, y = params.simulate(1000)
    ks, records = estimate_using_em_pso(y,
        estimates="F H Q R X0 P0",
        F0=params.F, H0=params.H, Q0=params.Q, R0=params.R, X00=params.X0, P00=params.P0,
        min_iterations=10, max_iterations=20, min_improvement=0.01, lat_dim=None,
        sample_size=10, population_size=100,
        #penalty_factors=(0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 0.5, 0.5)
    )
    neoparams = ks.parameters
    #print(records)
    #neoparams.show()
    #params.show()
    #params_orig.show()
    assert (np.abs(np.mean(neoparams.X0) - -50) <= 0.15 * 50), "Failed simulation: mean(X0 pred) != true mean"
    assert (np.abs(np.mean(neoparams.F) - 1) <= 0.15 * 1), "Failed simulation: mean(F pred) != true mean"
    assert (np.abs(np.mean(neoparams.H) - 10) <= 0.15 * 10), "Failed simulation: mean(H pred) != true mean"


if __name__ == "__main__":
    pass
    #import pytest
    #pytest.main(sys.argv[0])
    #_test_pure_pso_1()

# For testing run
# pip install pytest
# or
# py.test kalman.py

#chrome-extension://oemmndcbldboiebfnladdacbdfmadadm/https://people.eecs.berkeley.edu/~pabbeel/cs287-fa13/slides/Likelihood_EM_HMM_Kalman.pdf
