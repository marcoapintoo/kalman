import os
import sys
sys.path.append(os.path.realpath(os.path.dirname(sys.argv[0]) + "/../../../src/python/"))
sys.path.append(os.path.realpath(os.path.dirname(sys.argv[0]) + "/../"))
sys.path.append(os.path.realpath(os.path.dirname(sys.argv[0]) + "/../../../bin/"))
sys.path.append(os.path.realpath(os.path.dirname(sys.argv[0]) + "/../../../tests/python/module/"))
import pickle
import numpy as np
import state_space_model as ssm
import plot_utils

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__ 

class paramdict:
    F = None
    H = None
    Q = None
    H = None
    X0 = None
    P0 = None
    obs_dim = None
    lat_dim = None

    def copy(self):
        obj = paramdict()
        for name in ["F", "H", "Q", "R", "X0", "P0", "obs_dim", "lat_dim"]:
            setattr(obj, name, getattr(self, name))
        return obj

def _paramdict():
    return dotdict({
        "F": None,
        "H": None,
        "Q": None,
        "R": None,
        "X0": None,
        "P0": None,
        "obs_dim": None,
        "lat_dim": None,
    })

class resdict:
    Xp = None
    Pp = None
    Yp = None
    Xf = None
    Pf = None
    Yf = None
    Xs = None
    Ps = None
    Ys = None
    LL = None
    F = None
    H = None
    Q = None
    H = None
    X0 = None
    P0 = None
    obs_dim = None
    lat_dim = None

def _resdict():
    return dotdict({
        "Xp": None,
        "Pp": None,
        "Yp": None,
        "Xf": None,
        "Pf": None,
        "Yf": None,
        "Xs": None,
        "Ps": None,
        "Ys": None,
        "LL": None,
        #
        "F": None,
        "H": None,
        "Q": None,
        "R": None,
        "X0": None,
        "P0": None,
        "obs_dim": None,
        "lat_dim": None,
    })

def resdict_fromdata(V):
    obj = resdict()
    obj.F = V[0]
    obj.H = V[1]
    obj.Q = V[2]
    obj.R = V[3]
    obj.X0 = V[4]
    obj.P0 = V[5]
    #
    obj.Xp = V[6]
    obj.Pp = V[7]
    obj.Yp = V[8]
    obj.Xf = V[9]
    obj.Pf = V[10]
    obj.Yf = V[11]
    obj.Xs = V[12]
    obj.Ps = V[13]
    obj.Ys = V[14]
    obj.LL = V[15]
    #
    return obj

###############################################################################
# SIMULATE PARAMS
###############################################################################
ALL_PARAMETERS = "F H Q R X0 P0"

def simVAR(n, A, X0, P0):
    Xs = np.zeros((X0.shape[0], n), dtype="f8")
    Xs[:, 0] = X0.ravel()
    for t in range(1, n - 1):
        Xs[:, t] = A.dot(Xs[:, t - 1]) + np.random.multivariate_normal(np.zeros(P0.shape[0]), P0)[0]
    return Xs

def simObs(H, Xs, R):
    n = Xs.shape[1]
    Ys = np.zeros((H.shape[0], n), dtype="f8")
    for t in range(n):
        Ys[:, t] = H.dot(Xs[:, t - 1]) + np.random.multivariate_normal(np.zeros(R.shape[0]), R)[0]
    return Ys

def simSSM(n, F, H, X0, P0, Q, R):
    X0e = np.random.multivariate_normal(X0.ravel(), P0).reshape(X0.shape)
    Xs = simVAR(n, F, X0e, Q)
    Ys = simObs(H, Xs, R)
    return Xs, Ys

def simulateParams(n, params):
    return simSSM(n, params.F, params.H, params.X0, params.P0, params.Q, params.R)

###############################################################################
# PARAMETERS
###############################################################################
def create_true_params_1x3_true(T = 300):
    params = paramdict()
    params.F = np.array([
        [1 - 1e-7],
    ])
    params.H = np.array([
        [ -1], 
        [ 0.1], 
        [  1.0],
    ])
    params.Q = np.array([
        [  1e-5],   
    ])
    params.R = np.array([
        [  0.1, 0.07,  0.01],
        [0.07,     4,  0.01],
        [ 0.01,  0.01,   500],
    ])
    params.X0 = np.array([
        [70],
    ])
    params.P0 = np.array([
        [  1e-3 ],
    ])
    print("Stability of F:", np.linalg.eig(params.F)[0])
    params.obs_dim = 3
    params.lat_dim = 1
    np.random.seed(0)
    X, Y = simulateParams(T, params)
    return params, X, Y

def create_true_params_1x3(T = 300):
    params = paramdict()
    params.F = np.array([
        [1 - 1e-3],
    ])
    params.H = np.array([
        [ -1], 
        [ 0.1], 
        [  1.0],
    ])
    params.Q = np.array([
        [  100],
    ])
    params.R = np.array([
        [  100, 0.07,  0.01],
        [0.07,     4,  0.01],
        [ 0.01,  0.01,   500],
    ])
    params.X0 = np.array([
        [7],
    ])
    params.P0 = np.array([
        [  0.1 ],
    ])
    print("Stability of F:", np.linalg.eig(params.F)[0])
    params.obs_dim = 3
    params.lat_dim = 1
    np.random.seed(0)
    X, Y = simulateParams(T, params)
    return params, X, Y

def create_true_params_2x3(T = 300):
    params = paramdict()
    params.F = np.array([
        [99, 20],
        [-4, -80],
    ])/ 100
    params.H = np.array([
        [ -1, 1], 
        [ 0.5, -0.1], 
        [  1.5,  5.0],
    ])
    params.Q = np.array([
        [  10, 0.001],
        [0.001,     1],
    ])
    params.R = np.array([
        [  0.1, 0.007,  0.01],
        [0.007,     5,  0.01],
        [ 0.01,  0.01,   0.1],
    ])
    params.X0 = np.array([
        [87],
        [0],
    ])
    params.P0 = np.array([
        [  0.09, 0.001],
        [0.001,   0.15],
    ])
    print("Stability of F:", np.linalg.eig(params.F)[0])
    params.obs_dim = 3
    params.lat_dim = 2
    np.random.seed(0)
    X, Y = simulateParams(T, params)
    return params, X, Y

##simulate VAR as in Enders 2004, p 268
def create_true_params_enders_2x2_true(T = 300):
    params = paramdict()
    params.F = np.array([
        [0.7, 0.2],
        [0.2, 0.7],
    ])
    params.Q = np.array([
        [0.1, 0.01],
        [0.01, 0.1],
    ])
    params.X0 = np.array([100, 80]).reshape(-1, 1)
    params.P0 = np.array([
        [0.1, 0.01],
        [0.01, 0.1],
    ])
    params.H = np.array([
        [10.0, 0.0],
        [0.1, 1],
    ])
    params.R = np.array([
        [0.01, 0.01],
        [0.01, 0.01],
    ])
    print("Stability of F:", np.linalg.eig(params.F)[0])
    params.obs_dim = 2
    params.lat_dim = 2
    np.random.seed(0)
    X, Y = simulateParams(T, params)
    return params, X, Y

def noising_params(params, factor=0.2):
    np.random.seed(123456)
    noise_params = paramdict()
    noise_params.F = params.F + factor * np.random.randn(*params.F.shape)
    noise_params.H = params.H + factor * np.random.randn(*params.H.shape)
    noise_params.Q = params.Q + factor * np.random.randn(*params.Q.shape)
    noise_params.R = params.R + factor * np.random.randn(*params.R.shape)
    noise_params.X0 = params.X0 + factor * np.random.randn(*params.X0.shape)
    noise_params.P0 = params.P0 + factor * np.random.randn(*params.P0.shape)
    noise_params.lat_dim = params.lat_dim
    noise_params.obs_dim = params.obs_dim
    return noise_params


###############################################################################
# EVALUATION METRIC
###############################################################################
def mse(Xtrue, Xest):
    #return np.mean(np.abs((Xtrue.ravel() - Xest.ravel()) ** 2))
    return np.sqrt(np.mean((Xtrue.ravel() - Xest.ravel()) ** 2) / (np.mean(Xtrue.ravel()**2) * np.mean(Xest.ravel()**2) + 1e-100 ) )

###############################################################################
# NOISING THE TRUE PARAMETERS
###############################################################################
def create_noised_params(n, params, estimates, noise_factor, fname=None):
    if fname is not None and os.path.exists(fname):
        with open(fname, "rb") as f:
            random_params = pickle.load(f)
        return random_params 
    np.random.seed()
    random_params = []
    for _ in range(n):
        noised_params = params.copy()    
        for p in ALL_PARAMETERS.split():
            if p in estimates:
                setattr(noised_params, p, getattr(params, p) + noise_factor * np.random.randn(*getattr(params, p).shape))
                if p in ["Q", "R", "P0"]:
                    _fix_variance(getattr(noised_params, p))
        random_params.append((noised_params, estimates, noise_factor))

    if fname is not None:
        with open(fname, "wb") as f:
            pickle.dump(random_params, f)
    return random_params

def _fix_variance(V):
    V[np.diag_indices(V.shape[0])] = np.abs(np.diag(V))
    
def create_set_noised_parameters(params, fname=None):
    np.random.seed()
    if fname is not None and os.path.exists(fname):
        with open(fname, "rb") as f:
            random_params = pickle.load(f)
        return random_params
    noised_factors = (
        [(1, 0.0)]
        + [(10, 0.01 * x) for x in range(10)]
        + [(10, 0.1 * x) for x in range(10)]
        + [(10, 1 * x) for x in range(10)]
    )
    noised_factors = (
        [(1, 0.0)]
        + [(10, 0.1 * x) for x in range(5)]
    )
    random_params = [] 
    for qnt, factor in noised_factors:
        for estimates in (list(ALL_PARAMETERS.split()) + [ALL_PARAMETERS]):
            random_params += create_noised_params(qnt, params, estimates, factor, fname=None)
    if fname is not None:
        with open(fname, "wb") as f:
            pickle.dump(random_params, f)
    return random_params


###############################################################################
# TEST A SET ESTIMATORS
###############################################################################
def _test_set_parameters(type_estimator, X, Y, simparams):
    N = len(simparams)
    inErrs = np.zeros(N)
    outErrs = np.zeros(N)
    performance_metrics = np.zeros((9, N))
    bestOutErr = -1e100
    bestEstParams = None
    params = simparams[0][0]
    for i in range(N):
        estimates = simparams[i][1]
        current_params = simparams[i][0]
        """
        oper = dict(F=current_params.F, H=current_params.H, 
                     Q=current_params.Q, R=current_params.R, 
                     X0=current_params.X0, P0=current_params.P0)
        for k1, v1 in oper.items():
            print("*", k1, "=", v1.shape)
        print("*", "Y", "=", Y.shape)
        """
        current_data = ssm.estimate(
                     type_estimator, estimates, Y,
                     Y.shape[0], X.shape[0], Y.shape[1],
                     F=current_params.F, H=current_params.H, 
                     Q=current_params.Q, R=current_params.R, 
                     X0=current_params.X0, P0=current_params.P0, 
                     min_iterations=2,
                     max_iterations=50, 
                     min_improvement=0.01,
                     sample_size=3000,
                     population_size=100,
                     penalty_low_variance_Q=0.1,
                     penalty_low_variance_R=0.1,
                     penalty_low_variance_P0=0.1,
                     penalty_low_std_mean_ratio=0.1,
                     penalty_inestable_system=10.0,
                     penalty_mse=10,#1e-1,
                     penalty_roughness_X=0,#0.5,
                     penalty_roughness_Y=0,#0.5,
                     max_length_loglikelihood=1000,
                     return_details=True)
        current_data = resdict_fromdata(current_data)
        
        performance = ssm.performance_of_parameters(Y,
            Y.shape[0], X.shape[0], Y.shape[1], 
            current_data.F, current_data.H, 
            current_data.Q, current_data.R, 
            current_data.X0, current_data.P0)
        performance_metrics[:, i] = performance
        
        inErr, outErr = 0, 0
        all_valid_estimates = ALL_PARAMETERS.split()
        for p in all_valid_estimates:
            if p in estimates:
                inErr += mse(getattr(params, p), getattr(current_params, p))
                outErr += mse(getattr(params, p), getattr(current_data, p))
        inErr, outErr = inErr/len(estimates.split()), outErr/len(estimates.split())
        inErrs[i] = inErr
        outErrs[i] = outErr
        if outErr > bestOutErr:
            bestOutErr = outErr
            bestEstParams = current_data
        #print("o", end=" {0:.1f} ".format(np.mean(outErrs[:i+1])))
    #print()
    return inErrs, outErrs, bestEstParams, performance_metrics
    

###############################################################################
# EVALUATE ESTIMATOR
###############################################################################
def eval_estimator(type_estimator, X, Y, simparams, cachefile=None):
    all_estimates = (list(ALL_PARAMETERS.split()) + [ALL_PARAMETERS])

    if cachefile is not None and os.path.exists(cachefile):
        with open(cachefile, "rb") as f:
            results = pickle.load(f)
        for estimates in all_estimates:
            print("Estimates", estimates)
            X1 = np.round(results[estimates]["in"], 2)
            X2 = np.round(results[estimates]["out"], 5)
            X1 = results[estimates]["in"]
            X2 = results[estimates]["out"]
            #X1 = results[estimates]["factors"]
            plot_utils.plot_error_density(X1, X2, title="Estimation error of {}".format(estimates))
            plot_utils.plot_error_confidence_interval2(X1, X2, title="Estimation error of {}".format(estimates), maxerror=0.5);plt.show()
            ####plot_utils.plot_error_confidence_interval2(X1, results[estimates]["out"], title="Estimation error of {}".format(estimates))
            plot_utils.show()
        return results

    results = {}
    
    
    for estimates in all_estimates:
        simparams2 = [(a,b,c) for (a,b,c) in simparams if b == estimates]
        idxsimparams2 = [i for i, (a,b,c) in enumerate(simparams) if b == estimates]
        factors = np.array([c for (a,b,c) in simparams if b == estimates])
        print("Estimates", estimates)
        
                    
        inErrs, outErrs, bestEstParams, performance_metrics = _test_set_parameters(type_estimator, X, Y, simparams2)
        #'''
        # Remove outliers
        xperformance_metrics = performance_metrics.copy()
        xfactors = factors.copy()
        xinErrs = inErrs.copy()
        xoutErrs = outErrs.copy()
        
        performance_metrics = performance_metrics[:, np.isfinite(outErrs)]
        inErrs = inErrs[np.isfinite(outErrs)]
        factors = factors[np.isfinite(outErrs)]
        outErrs = outErrs[np.isfinite(outErrs)]
        
        #inErrs = inErrs[np.abs(outErrs - outErrs.mean()) < 7 * outErrs.std()]
        #factors = factors[np.abs(outErrs - outErrs.mean()) < 7 * outErrs.std()]
        #outErrs = outErrs[np.abs(outErrs - outErrs.mean()) < 7 * outErrs.std()]
        #
        try:
            X1 = np.round(inErrs, 2)
            X2 = np.round(outErrs, 5)
            X1 = inErrs
            X2 = outErrs
            plot_utils.plot_error_density(X1, X2, title="Estimation error of {}".format(estimates))
            plot_utils.plot_error_confidence_interval2(X1, X2, title="Estimation error of {}".format(estimates), maxerror=0.5);plt.show()
            ####plot_utils.plot_error_confidence_interval2(X1, outErrs, title="Estimation error of {}".format(estimates))
            plot_utils.show()
        except Exception as e:
            print("ERROR: Plot failed!", e)
            plot_utils.close()
            
        results[estimates] = {
            "in": inErrs, "out": outErrs, 
            "xin": xinErrs, "xout": xoutErrs, 
            "estimator": bestEstParams, "idx": np.array(idxsimparams2),
            "factors": factors, "xfactors": xfactors,
            "performance": performance_metrics, "xperformance": xperformance_metrics,
        }
    
    if cachefile is not None:
        with open(cachefile, "wb") as f:
            pickle.dump(results, f)
    return results

###############################################################################
## 
###############################################################################

def plot_error_confidence_interval3(inErrs, outErrs, name="X", maxerror=None, color="darkblue"):
    inErrs, outErrs = np.round(inErrs, 8), np.round(outErrs, 8)
    
    if maxerror is not None:
        outErrs = outErrs[inErrs <= maxerror]
        inErrs = inErrs[inErrs <= maxerror]
    
    mean, std, percentile = np.mean, np.std, lambda p: (lambda y: np.percentile(y, p*100, interpolation="linear"))
    
    def bootstrap(y, measure):
        sample_size = max(5, y.shape[0] // 10)
        ys = []
        for i in range(200):
            yb = np.random.choice(y, sample_size)
            ys.append(np.mean(yb))
        return measure(ys)
    
    # regroup series
    xinterval = np.sort(np.unique(inErrs))
    
    yinterval = []
    for i, x0 in enumerate(xinterval):
        yinterval.append(outErrs[inErrs == x0])
        if len(yinterval[i]) < 10:
            yinterval[i] = np.repeat(yinterval[i], 10)
        yinterval[i] = yinterval[i][np.abs(yinterval[i] - yinterval[i].mean()) <= 5 * yinterval[i].std()]
    
    # Remove empty spaces
    intervals = len(xinterval)
    xinterval = np.array([xinterval[i] for i in range(intervals) if len(yinterval[i]) >= 10])
    yinterval = [yinterval[i] for i in range(intervals) if len(yinterval[i]) >= 10]
    intervals = len(xinterval)
    
    yinterval_mean = np.array([bootstrap(yinterval[i], mean) for i in range(intervals)])
    yinterval_std = np.array([bootstrap(yinterval[i], std) for i in range(intervals)])
    yinterval_sup_ci = np.array([bootstrap(yinterval[i], percentile(0.95)) for i in range(intervals)])
    yinterval_inf_ci = np.array([bootstrap(yinterval[i], percentile(0.05)) for i in range(intervals)])
    
    plt.plot(inErrs[0], outErrs[0], "o", color=color, alpha=0, markersize=1, label="{2}".format(outErrs[0], outErrs.mean(), name))
    plt.plot(inErrs[0], outErrs[0], "o", color=color, alpha=1, markersize=1, label=" Global mean error : {0:.2g}".format(yinterval_mean.mean()))
    
    
    plt.plot(inErrs, outErrs, ".", color=color, alpha=0.3, markersize=3)
    plt.plot(xinterval, yinterval_mean, "-", color=color, label=" Mean error ".format(name))
    
    plt.plot(xinterval, yinterval_sup_ci, "-", color=color, linewidth=0.5, alpha=0.5, label=" Confidence interval".format(name))
    plt.plot(xinterval, yinterval_inf_ci, "-", color=color, linewidth=0.5, alpha=0.5)
    
    plt.fill_between(xinterval, yinterval_sup_ci, yinterval_inf_ci, color=color, alpha=0.1)
    
    #plt.plot(inErrs[0], outErrs[0], "o", color=color, alpha=1, markersize=10, label="Default error {2}: {0:.2g}\nGlobal error {2}: {1:.2g}".format(outErrs[0], outErrs.mean(), name))
    
    ptp_ci = yinterval_sup_ci.max() - yinterval_inf_ci.min()
    #plt.ylim(max(-0.2, yinterval_inf_ci.min() - 0.2 * ptp_ci), 0.2 * ptp_ci + yinterval_sup_ci.max())
    plt.xlim(xinterval.min(), xinterval.max())



def compare_estimation_errors(*set_results):
    all_estimates = (list(ALL_PARAMETERS.split()) + [ALL_PARAMETERS])
    max_error = 10.5

    for estimates in all_estimates:
        plt.figure(figsize=(15, 10))
        plt.title("Estimation error of {0}".format(estimates))
        #set_results = [results_em, results_pso, results_em_pso, results_lse_pso]
        set_colors = ["sienna", "darkcyan", "olivedrab", "steelblue"]
        set_names = ["EM", "PSO", "EM+PSO", "LSE+PSO"]
        for results, color, name in zip(set_results, set_colors, set_names):
            x = results[estimates]["xfactors"]
            y = results[estimates]["xout"]
            #x, y = x[x < max_error], y[x < max_error]
            plot_error_confidence_interval3(x, y, name=name, maxerror=max_error, color=color)
        plt.legend()
        plt.tight_layout()
        plt.show()

def compare_estimation_errors2(*set_results, ylims=None, set_names = ("EM", "PSO", "EM+PSO", "LSE+PSO")):
    all_estimates = ([ALL_PARAMETERS] + list(ALL_PARAMETERS.split()))
    axes = [
        lambda: plt.subplot2grid((4, 2), (0, 0), colspan=2),
        lambda: plt.subplot2grid((4, 2), (1, 0)),
        lambda: plt.subplot2grid((4, 2), (1, 1)),
        lambda: plt.subplot2grid((4, 2), (2, 0)),
        lambda: plt.subplot2grid((4, 2), (2, 1)),
        lambda: plt.subplot2grid((4, 2), (3, 0)),
        lambda: plt.subplot2grid((4, 2), (3, 1)),
    ]
    max_error = 1
    plt.figure(figsize=(25, 7.5*4))
    for n, estimates in enumerate(all_estimates):
        axes[n]()
        plt.title("Estimation error of {0}".format(estimates))
        #set_results = [results_em, results_pso, results_em_pso, results_lse_pso]
        set_colors = ["sienna", "darkcyan", "olivedrab", "steelblue"]
        for results, color, name in zip(set_results, set_colors, set_names):
            x = results[estimates]["xfactors"]
            y = results[estimates]["xout"]
            #x, y = x[x < max_error], y[x < max_error]
            plot_error_confidence_interval3(x, y, name=name, maxerror=max_error, color=color)
        if ylims is not None and ylims[n] is not None:
            plt.ylim(-0.2, ylims[n])
        plt.legend()
    plt.tight_layout()
    plt.show()



def load_data_results(*fnames):
    data_results = []
    for fname in fnames:
        with open(fname, "rb") as f:
            data_results.append(pickle.load(f))
    return data_results

###############################################################################
## 
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (15, 10)

def main():
    params1, X1, Y1 = create_true_params_enders_2x2_true()
    X1, Y1 = X1[:, :100], Y1[:, :100]
    X1, Y1 = X1[:, :10], Y1[:, :10]
    simparams1 = create_set_noised_parameters(params1, fname="random-params1")
    ##results_em = eval_estimator("em", X1, Y1, simparams1, cachefile="test1-em")
    ##results_pso = eval_estimator("pso", X1, Y1, simparams1, cachefile="test1-pso")
    #
    current_data = ssm.estimate(
        "PSO", "F H Q R X0 P0", Y1,
        Y1.shape[0], X1.shape[0], Y1.shape[1],
        F=None, H=None, 
        Q=None, R=None, 
        X0=None, P0=None, 
        min_iterations=2,
        max_iterations=50, 
        min_improvement=0.01,
        sample_size=3000,
        population_size=100,
        penalty_low_variance_Q=0.1,
        penalty_low_variance_R=0.1,
        penalty_low_variance_P0=0.1,
        penalty_low_std_mean_ratio=0.1,
        penalty_inestable_system=10.0,
        penalty_mse=100,#1e-1,
        penalty_roughness_X=0,#0.5,
        penalty_roughness_Y=0,#0.5,
        max_length_loglikelihood=1000,
        return_details=True)
    #params3.H, current_params.H, current_data.H
    print("F:", current_data[0])
    print("H:", current_data[1])

if __name__ == '__main__':
    main()

