import os
import sys
sys.path.append(os.path.realpath(os.path.dirname(sys.argv[0]) + "/../../../src/python/pure/"))
sys.path.append(os.path.realpath(os.path.dirname(sys.argv[0]) + "/../"))
import numpy as np
import kalman
kalman.resourceParams["trace.intermediates"] = False

from plot_utils import *

def plot_results(smoother, X=None, Y=None, predicted=False):
    Xs = smoother.Xs()
    Ys = smoother.Ys()
    Xp = smoother.Xp() if predicted else None
    Yp = smoother.Yp() if predicted else None
    plot_smoothed_predicted_data(Xs, Ys, Yp, Xp, X, Y)
    
###############################################################################
# PARAMETERS
###############################################################################
def create_true_params_1x3_true():
    params = kalman.SSMParameters()
    params.F = np.array([
        [1 - 1e-7],
    ])
    print(np.linalg.eig(params.F))
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
    params.set_dimensions(3, 1)
    np.random.seed(0)
    T = 300
    X, Y = params.simulate(T)
    return params, X, Y

def create_true_params_1x3():
    params = kalman.SSMParameters()
    params.F = np.array([
        [1 - 1e-3],
    ])
    print(np.linalg.eig(params.F))
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
    params.set_dimensions(3, 1)
    np.random.seed(0)
    T = 300
    X, Y = params.simulate(T)
    return params, X, Y

def create_true_params_2x3():
    params = kalman.SSMParameters()
    params.F = np.array([
        [99, 20],
        [-4, -80],
    ])/ 100
    print(np.linalg.eig(params.F))
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
        [  0.9, 0.001],
        [0.001,   0.5],
    ])
    params.set_dimensions(3, 2)
    np.random.seed(0)
    T = 200
    X, Y = params.simulate(T)
    return params, X, Y


def initial_parameters(type, factor_noise=0.2):
    if type == "1x3":
        params, X, Y = create_true_params_1x3()
    elif type == "1x3_true":
        params, X, Y = create_true_params_1x3_true()
    elif type == "2x3":
        params, X, Y = create_true_params_2x3()
    plot_data(X, Y)
    noised_params = noising_params(params, factor=factor_noise)
    print("****** TRUE  ******")
    params.show()
    print(params.performance_parameters_line(Y))
    print("****** FALSE  ******")
    noised_params.show()
    print(noised_params.performance_parameters_line(Y))
    return X, Y, params, noised_params

def noising_params(params, factor=0.2):
    np.random.seed(123456)
    noise_params = kalman.SSMParameters()
    noise_params.F = params.F + factor * np.random.randn(*params.F.shape)
    noise_params.H = params.H + factor * np.random.randn(*params.H.shape)
    noise_params.Q = params.Q + factor * np.random.randn(*params.Q.shape)
    noise_params.R = params.R + factor * np.random.randn(*params.R.shape)
    noise_params.X0 = params.X0 + factor * np.random.randn(*params.X0.shape)
    noise_params.P0 = params.P0 + factor * np.random.randn(*params.P0.shape)
    noise_params.set_dimensions(3, 2)
    return noise_params


###############################################################################
# TEST BASE
###############################################################################
def test_metaestimator(type_estimator, X, Y, params, estimates="F H Q R X0 P0", sample_size=3000, min_iterations=5, max_iterations=20, min_improvement=0.01, population_size=10, penalty_factors={}, plot_intermediate_results=True):
    neo_params = params.copy()
    neo_params.Q = np.eye(*neo_params.Q.shape) * 0.1
    neo_params.R = np.eye(*neo_params.R.shape) * 0.1
    neo_params.P0 = np.eye(*neo_params.P0.shape) * 0.1
    neo_params2, _ = test_estimator(
        type_estimator,
        X, Y, neo_params,
        estimates=' '.join(w for w in ["F", "H", "X0"] if w in estimates),
        sample_size=sample_size, 
        min_iterations=min_iterations,
        max_iterations=max_iterations, 
        min_improvement=min_improvement, 
        population_size=population_size, 
        penalty_factors=penalty_factors,
        plot_intermediate_results=plot_intermediate_results,
    )
    neo_params2 = neo_params2.parameters
    neo_params3, _ = test_estimator(
        type_estimator,
        X, Y, neo_params2,
        estimates=' '.join(w for w in ["Q", "R", "P0"] if w in estimates),
        sample_size=sample_size, 
        min_iterations=min_iterations,
        max_iterations=max_iterations, 
        min_improvement=min_improvement, 
        population_size=population_size, 
        penalty_factors=penalty_factors,
        plot_intermediate_results=plot_intermediate_results,
    )
    neo_params3 = neo_params3.parameters
    return test_estimator(
        type_estimator,
        X, Y, neo_params3,
        estimates=estimates,
        sample_size=sample_size, 
        min_iterations=min_iterations,
        max_iterations=max_iterations, 
        min_improvement=min_improvement, 
        population_size=population_size, 
        penalty_factors=penalty_factors,
        plot_intermediate_results=plot_intermediate_results,
    )

def test_estimator(type_estimator, X, Y, params, estimates="F H Q R X0 P0", sample_size=3000, min_iterations=5, max_iterations=20, min_improvement=0.01, population_size=10, penalty_factors={}, plot_intermediate_results=True):
    np.random.seed(42)

    estimators = {
        "em": kalman.estimate_using_em,
        "pso": kalman.estimate_using_pso,
        "lse+pso": kalman.estimate_using_lse_pso,
        "em+pso": kalman.estimate_using_em_pso,
    }

    pso_args = dict(
        #lat_dim=None,
        sample_size=sample_size,
        population_size=population_size,
        penalty_factors={
            "low_std_mean_ratio": penalty_factors.get("low_std_mean_ratio", 0.01),
            "low_variance_Q": penalty_factors.get("low_variance_Q", 0.05),
            "low_variance_R": penalty_factors.get("low_variance_R", 0.05),
            "low_variance_P0": penalty_factors.get("low_variance_P0", 0.05),
            "inestable_system": penalty_factors.get("inestable_system", 10),
            "mse": penalty_factors.get("mse", 10),
            "roughness_X": penalty_factors.get("roughness_X", 0.05),
            "roughness_Y": penalty_factors.get("roughness_Y", 0.05),
        },
    )

    f_estimator = estimators[type_estimator]

    if "pso" not in type_estimator:
        pso_args = {}

    smoother, record = f_estimator(
        Y=Y,
        estimates=estimates,
        F0=params.F.copy(),
        H0=params.H.copy(),
        Q0=params.Q.copy(),
        R0=params.R.copy(),
        X00=params.X0.copy(),
        P00=params.P0.copy(),
        min_iterations=min_iterations,
        max_iterations=max_iterations,
        min_improvement=min_improvement,
        **pso_args
    )

    if plot_intermediate_results:
        print(smoother.parameters.performance_parameters_line(Y))
        smoother.parameters.show()
        plot_likelihood_record(record)
        plot_results(smoother, X, Y)
    return smoother, record

#
#
#
def simVAR(n, A, X0, P0):
    # X1 = A X0 + N(0, P0)
    Xs = [X0]
    for i in range(n - 1):
        #print(Xs)
        Xs.append(A.dot(Xs[-1]) + np.random.multivariate_normal(np.zeros(P0.shape[0]), P0)[0])
    return np.array(Xs) # first: X[0]
    #return np.array(Xs).reshape(-1, X0.shape[1], X0.shape[0]).T # first: X[:, :, 0]

def simObs(F, Xs, R):
    Ys = []
    for t in range(Xs.shape[0]):
        Ys.append(F.dot(Xs[t]) + np.random.multivariate_normal(np.zeros(R.shape[0]), R)[0])
    return np.array(Ys) # first: Y[0]
    #return np.array(Ys).reshape(-1, Xs.shape[2], Xs.shape[1]).T # first: X[:, :, 0]

def simSSM(n, F, H, X0, P0, Q, R):
    X0e = np.random.multivariate_normal(X0.ravel(), P0).reshape(X0.shape)
    Xs = simVAR(n, F, X0e, Q)
    Ys = simObs(F, Xs, R)
    return Xs, Ys

#
#
#
def mse(Xtrue, Xest):
    #return np.mean(np.abs((Xtrue.ravel() - Xest.ravel()) ** 2))
    return np.sqrt(np.mean((Xtrue.ravel() - Xest.ravel()) ** 2))

import pickle
def createSamplesSingleParam(n, params, estimates, noise_factor, fname=None):
    if fname is not None and os.path.exists(fname):
        with open(fname, "rb") as f:
            random_params = pickle.load(f)
        return random_params 
    np.random.seed()
    random_params = []
    for _ in range(n):
        noised_params = params.copy()    
        for p in "F H Q R X0 P0".split():
            if p in estimates:
                setattr(noised_params, p, getattr(params, p) + noise_factor * np.random.randn(*getattr(params, p).shape))
        random_params.append((noised_params, estimates, noise_factor))

    if fname is not None:
        with open(fname, "wb") as f:
            pickle.dump(random_params, f)
    return random_params

def createSampleNoisedParams(params, fname=None):
    np.random.seed()
    if fname is not None and os.path.exists(fname):
        with open(fname, "rb") as f:
            random_params = pickle.load(f)
        return random_params
    noised_factors = [
        (1, 0.0),
        (50, 0.01),
        (300, 0.1),
        (50, 1),
        (10, 1),
    ]
    noised_factors1 = [(1, 0.0),
        (1, 0.01),
        (1, 0.1)
    ]  
    random_params = [] 
    for qnt, factor in noised_factors:
        for estimates in (list("F H Q R X0 P0".split()) + ["F H Q R X0 P0"]):
            random_params += createSamplesSingleParam(qnt, params, estimates, factor, fname=None)
    if fname is not None:
        with open(fname, "wb") as f:
            pickle.dump(random_params, f)
    return random_params


def simParam2(type_estimator, X, Y, simparams, prevInErrs=[], prevOutErrs=[]):
    N = len(simparams)
    inErrs = np.zeros(N)
    outErrs = np.zeros(N)
    bestOutErr = -1e100
    bestEstParams = None
    params = simparams[0][0]
    for i in range(N):
        estimates = simparams[i][1]
        smt, _ = test_estimator(
            type_estimator,
            X, Y, simparams[i][0],
            estimates=estimates,
            sample_size=3000,
            population_size=10,
            min_iterations=2,
            max_iterations=30,
            penalty_factors={
                "mse": 1e-1,
            },
            plot_intermediate_results=False,
        )
        
        inErr, outErr = 0, 0
        p_estimates = "F H Q R X0 P0".split()
        for p in p_estimates:
            if p in estimates:
                inErr += mse(getattr(params, p), getattr(simparams[i][0], p))
                outErr += mse(getattr(params, p), getattr(smt.parameters, p))
        inErr, outErr = inErr/len(estimates.split()), outErr/len(estimates.split())        
        inErrs[i] = inErr
        outErrs[i] = outErr
        if outErr > bestOutErr:
            bestOutErr = outErr
            bestEstParams = smt.parameters
        print("o", end=" {0:.1f} ".format(np.mean(outErrs[:i+1])))
    print()
    return np.concatenate([prevInErrs, inErrs]), np.concatenate([prevOutErrs, outErrs]), bestEstParams
    
def simSingleParam(type_estimator, X, Y, params, estimates, noise_factor):
    np.random.seed()
    
    noised_params = params.copy()
    lstestimates = ""
    
    for p in "F H Q R X0 P0".split():
        if p in estimates:
            setattr(noised_params, p, getattr(params, p) + noise_factor * np.random.randn(*getattr(params, p).shape))
            #setattr(noised_params, p,
            #        getattr(params, p) + 
            #        noise_factor * np.random.randn() * np.ones(getattr(params, p).shape)
            #)
    
    #params.show()
    #noised_params.show()
    
    #test_metaestimator(
    smt, _ = test_estimator(
        type_estimator,
        X, Y, noised_params,
        estimates=estimates,
        sample_size=3000,
        population_size=10,
        min_iterations=5,
        max_iterations=30,
        penalty_factors={
            "mse": 1e-1,
        },
        plot_intermediate_results=False,
    )
    
    inErr, outErr = 0, 0
    for p in "F H Q R X0 P0".split():
        if p in estimates:
            inErr += mse(getattr(params, p), getattr(noised_params, p))
            outErr += mse(getattr(params, p), getattr(smt.parameters, p))
    '''
    print("===>", inErr)
    print("    ", outErr)
    print("    ", params.F)
    print("    ", noised_params.F)
    print("    ", smt.parameters.F)
    print(" == ", mse(params.F, noised_params.F))
    print("    ", mse(params.F, smt.parameters.F))
    '''
    return inErr, outErr, smt.parameters

def simParam(N, type_estimator, X, Y, params, estimates, noise_factor, prevInErrs=[], prevOutErrs=[]):
    inErrs = np.zeros(N)
    outErrs = np.zeros(N)
    bestOutErr = -1e100
    bestEstParams = None
    for i in range(N):
        inErr, outErr, est_params = simSingleParam(type_estimator, X, Y, params, estimates, noise_factor)
        inErrs[i] = inErr
        outErrs[i] = outErr
        if outErr > bestOutErr:
            bestOutErr = outErr
            bestEstParams = est_params
        print("o", end=" {0:.1f} ".format(np.mean(outErrs[:i+1])))
    print()
    #bestEstParams.show()
    return np.concatenate([prevInErrs, inErrs]), np.concatenate([prevOutErrs, outErrs]), bestEstParams

#
#
#
import pickle
import os
def test_default_estimates(type_estimator, X1, Y1, params, all_estimates="F H X0 P0 Q R", everything=False, cachefile=None):
    #all_estimates = "F H X0 P0 Q R"
    #type_estimator = "em"

    if cachefile is not None and os.path.exists(cachefile):
        with open(cachefile, "rb") as f:
            results = pickle.load(f)
        for estimates in (all_estimates.split() if not everything else [all_estimates]):
            print("Estimates", estimates)
            inErrs, outErrs = results["in"], results["out"]
            plot_error_density(inErrs, outErrs, title="Estimation error of {}".format(estimates))
            plot_error_confidence_interval(inErrs, outErrs, title="Estimation error of {}".format(estimates))
            plt.show()
        return results

    results = {}

    for estimates in (list(all_estimates.split()) + [all_estimates]):
        print("Estimates", estimates)
        inErrs, outErrs, _ = simParam(
            N=1,
            type_estimator=type_estimator,
            X=X1, Y=Y1,
            params=params.copy(),
            estimates=estimates,
            noise_factor=0.0,
        )
        inErrs, outErrs, bestEstParams = simParam(
            N=10*5,
            type_estimator=type_estimator,
            X=X1, Y=Y1,
            params=params.copy(),
            estimates=estimates,
            noise_factor=0.01,
        )
        inErrs, outErrs, _ = simParam(
            N=100*3,
            type_estimator=type_estimator,
            X=X1, Y=Y1,
            params=params.copy(),
            estimates=estimates,
            noise_factor=0.1,
            prevInErrs=inErrs,
            prevOutErrs=outErrs,
        )
        inErrs, outErrs, _ = simParam(
            N=20*5,
            type_estimator=type_estimator,
            X=X1, Y=Y1,
            params=params.copy(),
            estimates=estimates,
            noise_factor=1,
            prevInErrs=inErrs,
            prevOutErrs=outErrs,
        )
        inErrs, outErrs, _ = simParam(
            N=5*5,
            type_estimator=type_estimator,
            X=X1, Y=Y1,
            params=params.copy(),
            estimates=estimates,
            noise_factor=10,
            prevInErrs=inErrs,
            prevOutErrs=outErrs,
        )
        #'''
        # Remove outliers
        xinErrs = inErrs.copy()
        xoutErrs = outErrs.copy()
        inErrs[np.isnan(inErrs)] = 1e100
        outErrs[np.isnan(outErrs)] = 1e100
        inErrs = inErrs[np.abs(inErrs - inErrs.mean()) < 50 * inErrs.std()]
        outErrs = outErrs[np.abs(inErrs - inErrs.mean()) < 50 * inErrs.std()]
        #
        plot_error_density(inErrs, outErrs, title="Estimation error of {}".format(estimates))
        plot_error_confidence_interval(inErrs, outErrs, title="Estimation error of {}".format(estimates))
        plt.show()
        results[estimates] = {"in": inErrs, "out": outErrs, "xin": xinErrs, "xout": xoutErrs, "estimator": bestEstParams}
    
    if cachefile is not None:
        with open(cachefile, "wb") as f:
            pickle.dump(results, f)
    return results



def eval_estimator(type_estimator, X, Y, simparams, cachefile=None):
    all_estimates = (list("F H X0 P0 Q R".split()) + ["F H Q R X0 P0"])

    if cachefile is not None and os.path.exists(cachefile):
        with open(cachefile, "rb") as f:
            results = pickle.load(f)
        for estimates in all_estimates:
            print("Estimates", estimates)
            inErrs, outErrs = results[estimates]["in"], results[estimates]["out"]
            plot_error_density(inErrs, outErrs, title="Estimation error of {}".format(estimates))
            plot_error_confidence_interval(inErrs, outErrs, title="Estimation error of {}".format(estimates))
            plt.show()
        return results

    results = {}
    
    
    for estimates in all_estimates:
        simparams2 = [(a,b,c) for (a,b,c) in simparams if b == estimates]
        idxsimparams2 = [i for i, (a,b,c) in enumerate(simparams) if b == estimates]
        print("Estimates", estimates)
        inErrs, outErrs, bestEstParams = simParam2(type_estimator, X, Y, simparams2)
        #'''
        # Remove outliers
        xinErrs = inErrs.copy()
        xoutErrs = outErrs.copy()
        inErrs[np.isnan(inErrs)] = 1e100
        outErrs[np.isnan(outErrs)] = 1e100
        inErrs = inErrs[np.abs(inErrs - inErrs.mean()) < 50 * inErrs.std()]
        outErrs = outErrs[np.abs(inErrs - inErrs.mean()) < 50 * inErrs.std()]
        #
        try:
            plot_error_density(inErrs, outErrs, title="Estimation error of {}".format(estimates))
            plot_error_confidence_interval(inErrs, outErrs, title="Estimation error of {}".format(estimates))
            plt.show()
        except Exception as e:
            print("ERROR: Plot failed!", e)
            [(plt.clf(), plt.close()) for _ in range(4)]
        results[estimates] = {"in": inErrs, "out": outErrs, "xin": xinErrs, "xout": xoutErrs, "estimator": bestEstParams, "idx": np.array(idxsimparams2)}
    
    if cachefile is not None:
        with open(cachefile, "wb") as f:
            pickle.dump(results, f)
    return results

"""
%load_ext rpy2.ipython
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
%%R -o series,estimated -w 15 -h 10 --units in -r 200

library(tsDyn)
##simulate VAR as in Enders 2004, p 268
B1 <- matrix(c(0.7, 0.2, 0.2, 0.7), 2)
series <- VAR.sim(B=B1,n=100,include="none")
#ts.plot(var1, type="l", col=c(1,2))
estimated <- lineVar(series, lag=1, include="none")
"""
def create_true_params_enders_2x2_true():
    params = kalman.SSMParameters()
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
    params.set_dimensions(2, 2)
    np.random.seed(0)
    T = 300
    X, Y = params.simulate(T)
    return params, X, Y
    """
    params = kalman.SSMParameters()
    n = 50 * 2
    #n = 10
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
        [0.01, 0.001],
        [0.001, 0.01],
    ])
    print("Stability of F:", np.linalg.eig(params.F)[0])
    np.random.seed()
    X, Y = simSSM(n=n, F=params.F, Q=params.Q, X0=params.X0, P0=params.P0, H=params.H, R=params.R)
    X1, Y1 = X.reshape(-1, X.shape[1]).T, Y.reshape(-1, Y.shape[1]).T
    """
