import os
import sys
sys.path.append(
    os.path.join(
        os.path.realpath(os.path.dirname(sys.argv[0])), "../../../src/python/pure/"
    )
)
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (7.5, 5)
import kalman
kalman.resourceParams["trace.intermediates"] = False

def plot_likelihood_record(records):
    plt.figure(figsize=(15, 3))
    plt.plot(records, "-", color="red")
    plt.plot(records, "o", color="black")
    plt.xlabel("Iterations")
    plt.xlabel("Loglikelihood")
    plt.title("Loglikelihood improvements")
    
def plot_results(smoother, X=None, Y=None, predicted=False):
    Xs = smoother.Xs()
    Ys = smoother.Ys()
    Xp = smoother.Xp()
    Yp = smoother.Yp()
    plot_data(Xs, Ys, Yp, Xp, X, Y, predicted)
    
def plot_data(Xs, Ys, Yp=None, Xp=None, X=None, Y=None, predicted=False):
    nrows = max(Xs.shape[0], Ys.shape[0])
    plt.figure(figsize=(15, 5 * nrows))
    labels = ["Kalman smoothed signal"]
    if predicted:
        labels = ["Kalman predicted signal"] + labels
    if X is not None or Y is not None:
        labels = labels + ["Input signal"]
    for k in range(Xs.shape[0]):
        plt.subplot(nrows, 2, 2 * k + 2)
        if predicted:
            plt.plot(Xp[k, :].ravel(), linestyle="-", color="darkolivegreen")
            plt.title("Predicted input (dim. {0})".format(k))
        plt.plot(Xs[k, :].ravel(), linestyle="-", color="coral")
        plt.title("System input (dim. {0})".format(k + 1))
        if X is not None:
            plt.plot(X[k, :].ravel(), linestyle="-.", color="black", alpha=0.5)
        plt.legend(labels)
    for k in range(Ys.shape[0]):
        plt.subplot(nrows, 2, 2 * k + 1)
        if predicted:
            plt.plot(Yp[k, :].ravel(), linestyle="-", color="darkolivegreen")
            plt.title("Predicted input (dim. {0})".format(k))
        plt.plot(Ys[k, :].ravel(), "-", color="coral")
        plt.title("System output (dim. {0})".format(k + 1))
        if Y is not None:
            plt.plot(Y[k, :].ravel(), "-.", color="black", alpha=0.5)
        plt.legend(labels)
    plt.tight_layout()
    plt.show()


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
        [  100, 0.07,  0.01],
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
def test_metaestimator(type_estimator, X, Y, params, estimates="F H Q R X0 P0", sample_size=3000, min_iterations=5, max_iterations=20, min_improvement=0.01, population_size=10, penalty_factors={}):
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
        penalty_factors=penalty_factors
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
        penalty_factors=penalty_factors
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
        penalty_factors=penalty_factors
    )

def test_estimator(type_estimator, X, Y, params, estimates="F H Q R X0 P0", sample_size=3000, min_iterations=5, max_iterations=20, min_improvement=0.01, population_size=10, penalty_factors={}):
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

    print(smoother.parameters.performance_parameters_line(Y))
    smoother.parameters.show()
    plot_likelihood_record(record)
    plot_results(smoother, X, Y)
    return smoother, record


'''
X, Y, params, noised_params = initial_parameters()
test_estimator("em",
    X, Y, params,
    estimates="F H Q R X0 P0",
    sample_size=3000,
    penalty_factors={}
)
'''