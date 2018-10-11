import os
import sys
import pickle
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (7.5, 5)
import seaborn as sns

###############################################################################
# ADD PROJECT PATH 
###############################################################################
sys.path.append(
    os.path.join(
        os.path.realpath(os.path.dirname(sys.argv[0])), "../../../src/python/pure/"
    )
)
import kalman
kalman.resourceParams["trace.intermediates"] = False

###############################################################################
# PLOTTING FUNCTIONS
###############################################################################
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

def plot_simple_data(Xs, Ys):
    nrows = max(Xs.shape[0], Ys.shape[0])
    plt.figure(figsize=(15, 5 * nrows))
    for k in range(Xs.shape[0]):
        plt.subplot(nrows, 2, 2 * k + 2)
        plt.plot(Xs[k, :].ravel(), linestyle="-", color="coral")
        plt.title("System input (dim. {0})".format(k + 1))
        plt.legend(["Input signal"])
    for k in range(Ys.shape[0]):
        plt.subplot(nrows, 2, 2 * k + 1)
        plt.plot(Ys[k, :].ravel(), "-", color="coral")
        plt.title("System output (dim. {0})".format(k + 1))
        plt.legend(["Output signal"])
    plt.tight_layout()
    plt.show()

def plot_smoothers(smoothers):
    Xs = smoothers[0][1].Xs()
    Ys = smoothers[0][1].Ys()
    nrows = max(Xs.shape[0], Ys.shape[0])
    plt.figure(figsize=(15, 5 * nrows))
    for k in range(Xs.shape[0]):
        plt.subplot(nrows, 2, 2 * k + 2)
        plt.title("System input (dim. {0})".format(k + 1))
        labels = []
        for name, smoother in smoothers:
            labels.append("{0}".format(name))
            plt.plot(smoother.Xs()[k, :].ravel(), linestyle="-")
        plt.legend(labels)
    for k in range(Ys.shape[0]):
        plt.subplot(nrows, 2, 2 * k + 1)
        plt.title("System output (dim. {0})".format(k + 1))
        labels = []
        for name, smoother in smoothers:
            labels.append("{0}".format(name))
            plt.plot(smoother.Ys()[k, :].ravel(), linestyle="-")
        plt.legend(labels)
    plt.tight_layout()
    plt.show()


###############################################################################
# PLOTTING FUNCTIONS FOR ERRORS 
###############################################################################
def plot_error_density(inErrs, outErrs, title="Errors"):
    #sns.set_style("deep")
    plt.style.use("seaborn-deep")
    plt.figure(figsize=(10, 10))
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.set_xticks([])
    axHisty.set_yticks([])

    # the scatter plot:
    #axScatter.scatter(x, y)
    axScatter.plot(inErrs, outErrs, "o")

    # now determine nice limits by hand:
    #axScatter.set_xlim(xlim)
    #axScatter.set_ylim(ylim)

    sns.kdeplot(inErrs, bw="scott", kernel="triw", shade=True, ax=axHistx)
    sns.kdeplot(outErrs, bw='scott', kernel="triw", shade=True, ax=axHisty, vertical=True)

    qntts, bins = np.histogram(outErrs, bins=100)
    tentative_modes_y = (bins[:-1] + 0.5 * (bins[1] - bins[0]))[qntts > 10 * qntts.mean()]
    for y0 in tentative_modes_y:
        axHisty.axhline(y=y0, color="darkblue")
        axHisty.text(
            0, y0 + (bins[1]-bins[0])*0.9,
            "Mode: {0:.2g}".format(y0),
            color="darkblue"
        )
    y0 = outErrs.mean()
    axHisty.axhline(y=y0, color="darkred")
    axHisty.text(
        0, y0 + (bins[1]-bins[0])*0.9,
        "Mean: {0:.2g}".format(y0),
        color="darkred"
    )
    #axHisty.axvline(x=)
    qntts, bins = np.histogram(inErrs, bins=100)
    tentative_modes_x = (bins[:-1] + 0.5 * (bins[1] - bins[0]))[qntts > 10 * qntts.mean()]
    for x0 in tentative_modes_x:
        axHistx.axvline(x=x0, color="darkblue")
        axHistx.text(
            x0 + 0.01 * (axHistx.get_xlim()[1] - axHistx.get_xlim()[0]), 
        0.45 * (axHistx.get_ylim()[1] - axHistx.get_ylim()[0]), 
            "Mode: {0:.2g}".format(y0),
            color="darkblue"
        )
    x0 = inErrs.mean()
    axHistx.axvline(x=x0, color="darkred")
    axHistx.text(
        x0 + 0.01 * (axHistx.get_xlim()[1] - axHistx.get_xlim()[0]), 
        0.45 * (axHistx.get_ylim()[1] - axHistx.get_ylim()[0]), 
        "Mean: {0:.2g}".format(x0),
        color="darkred"
    )

    axScatter.set_xlim(axHistx.get_xlim())
    axScatter.set_ylim(axHisty.get_ylim())
    
    #axHistx.set_ylabel("")
    axScatter.set_ylabel("MSE of input parameter")
    
    #axHisty.set_xlabel("")
    axScatter.set_xlabel("MSE of output parameter")
    
    axHistx.set_title(" "* 40 + title)
    #plt.suptitle(title)    

def plot_error_confidence_interval(inErrs, outErrs, intervals = 50, title="Estimation error of X"):
    inErrs, outErrs = np.round(inErrs, 5), np.round(outErrs, 5)
    plt.figure(figsize=(15, 10))
    plt.title(title)
    def bootstrap(y, measure):
        sample_size = max(5, y.shape[0] // 10)
        ys = []
        for i in range(200):
            yb = np.random.choice(y, sample_size)
            ys.append(np.mean(yb))
        return measure(ys)

    intervals = min(intervals, inErrs.shape[0] // 10)
    #print(intervals)
    xinterval = np.linspace(inErrs.min(), inErrs.max(), intervals + 1)
    delta_xinterval = xinterval[1] - xinterval[0]
    yinterval = []
    inErrsRounded = inErrs.copy()
    for i in range(intervals):
        yinterval.append(
            outErrs[(xinterval[i + 1] >= inErrs) * (inErrs >= xinterval[i])]
        )
        inErrsRounded[(xinterval[i + 1] >= inErrs) * (inErrs >= xinterval[i])] = xinterval[i] + 0.5 * delta_xinterval
    xinterval = xinterval[:-1] + 0.5 * delta_xinterval
    
    # Remove empty spaces
    xinterval = np.array([o for k, o in enumerate(xinterval) if len(yinterval[k]) > 5])
    yinterval = [o for o in yinterval if len(o) > 5]
    intervals = len(xinterval)
    
    #yinterval_mean = np.array([np.mean(yinterval[i]) for i in range(intervals)])
    yinterval_mean = np.array([bootstrap(yinterval[i], np.mean) for i in range(intervals)])
    #yinterval_std = np.array([np.std(yinterval[i]) for i in range(intervals)])
    yinterval_std = np.array([bootstrap(yinterval[i], np.std) for i in range(intervals)])
    yinterval_sup_ci = np.array([bootstrap(yinterval[i], lambda y: np.percentile(y, 95, interpolation="linear")) for i in range(intervals)])
    yinterval_inf_ci = np.array([bootstrap(yinterval[i], lambda y: np.percentile(y, 5, interpolation="linear")) for i in range(intervals)])
    xinterval.shape, yinterval_mean.shape
    plt.plot(inErrsRounded, outErrs, ".", color="darkblue", alpha=0.3, markersize=3)
    plt.plot(xinterval, yinterval_mean, "-", color="darkblue", label="Mean error")
    plt.plot(xinterval, yinterval_mean + yinterval_std, "--", color="darkred", linewidth=0.5, label="Mean error +/- std. dev.", )
    plt.plot(xinterval, yinterval_mean - yinterval_std, "--", color="darkred", linewidth=0.5, )
    plt.plot(xinterval, yinterval_sup_ci, "-", color="darkred", linewidth=1, label="Confidence interval")
    plt.plot(xinterval, yinterval_inf_ci, "-", color="darkred", linewidth=1)
    #plt.fill_between(xinterval, yinterval_mean + 2 * yinterval_std, yinterval_mean -  2 * yinterval_std, color="red", alpha=0.1)
    plt.fill_between(xinterval, yinterval_mean + yinterval_std, yinterval_mean -  yinterval_std, color="red", alpha=0.05)
    plt.fill_between(xinterval, yinterval_sup_ci, yinterval_inf_ci, color="red", alpha=0.1)

    for i in range(intervals):
        y_errs = yinterval[i]
        factor_y = plt.gca().get_ylim()
        factor_x = plt.gca().get_xlim()
        x0 = xinterval[i] + (factor_x[1] - factor_x[0]) * 0.001
        dx0 = (factor_x[1] - factor_x[0]) * 0.01
        scale_plot = lambda y, factor=factor_y: (y - factor[0])/(factor[1] - factor[0])
        plt.axvline(x=x0, ymin=scale_plot(y_errs.min()), ymax=scale_plot(y_errs.max()), color="gray", alpha=0.5)
        plt.axhline(y=y_errs.min(), xmin=scale_plot(x0 - dx0, factor_x), xmax=scale_plot(x0 + dx0, factor_x), color="gray", alpha=1)
        plt.axhline(y=y_errs.max(), xmin=scale_plot(x0 - dx0, factor_x), xmax=scale_plot(x0 + dx0, factor_x), color="gray", alpha=1)
    #plt.plot(xinterval, yinterval, ".")
    plt.legend()



def plot_error_confidence_interval2(inErrs, outErrs, title="Estimation error of X", show_bars=False, maxerror=None):
    inErrs, outErrs = np.round(inErrs, 8), np.round(outErrs, 8)
    if maxerror is not None:
        outErrs = outErrs[inErrs <= maxerror]
        inErrs = inErrs[inErrs <= maxerror]
    plt.figure(figsize=(15, 10))
    plt.title(title)
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
    delta_xinterval = xinterval[1] - xinterval[0]
    yinterval = []
    for i, x0 in enumerate(xinterval):
        yinterval.append(outErrs[inErrs == x0])
        if len(yinterval[i]) < 10:
            yinterval[i] = np.repeat(yinterval[i], 10)
    
    # Remove empty spaces
    intervals = len(xinterval)
    yinterval_mean = np.array([bootstrap(yinterval[i], mean) for i in range(intervals)])
    yinterval_std = np.array([bootstrap(yinterval[i], std) for i in range(intervals)])
    yinterval_sup_ci = np.array([bootstrap(yinterval[i], percentile(0.95)) for i in range(intervals)])
    yinterval_inf_ci = np.array([bootstrap(yinterval[i], percentile(0.05)) for i in range(intervals)])
    
    
    
    plt.plot(inErrs, outErrs, ".", color="darkblue", alpha=0.3, markersize=3)
    plt.plot(xinterval, yinterval_mean, "-", color="darkblue", label="Mean error")
    plt.plot(xinterval, yinterval_mean + yinterval_std, "--", color="darkred", linewidth=0.5, label="Mean error +/- std. dev.", )
    plt.plot(xinterval, yinterval_mean - yinterval_std, "--", color="darkred", linewidth=0.5, )
    plt.plot(xinterval, yinterval_sup_ci, "-", color="darkred", linewidth=1, label="Confidence interval")
    plt.plot(xinterval, yinterval_inf_ci, "-", color="darkred", linewidth=1)
    #plt.fill_between(xinterval, yinterval_mean + 2 * yinterval_std, yinterval_mean -  2 * yinterval_std, color="red", alpha=0.1)
    plt.fill_between(xinterval, yinterval_mean + yinterval_std, yinterval_mean -  yinterval_std, color="red", alpha=0.05)
    plt.fill_between(xinterval, yinterval_sup_ci, yinterval_inf_ci, color="red", alpha=0.1)
    
    plt.plot(inErrs[0], outErrs[0], "o", color="darkblue", alpha=1, markersize=10, label="Default error: {0:.2g}\nGlobal error: {1:.2g}".format(outErrs[0], outErrs.mean()))
    
    if show_bars:
        for i in range(intervals):
            y_errs = yinterval[i]
            factor_y = plt.gca().get_ylim()
            factor_x = plt.gca().get_xlim()
            x0 = xinterval[i] + (factor_x[1] - factor_x[0]) * 0.001
            dx0 = (factor_x[1] - factor_x[0]) * 0.01
            scale_plot = lambda y, factor=factor_y: (y - factor[0])/(factor[1] - factor[0])
            plt.axvline(x=x0, ymin=scale_plot(y_errs.min()), ymax=scale_plot(y_errs.max()), color="gray", alpha=0.5)
            plt.axhline(y=y_errs.min(), xmin=scale_plot(x0 - dx0, factor_x), xmax=scale_plot(x0 + dx0, factor_x), color="gray", alpha=1)
            plt.axhline(y=y_errs.max(), xmin=scale_plot(x0 - dx0, factor_x), xmax=scale_plot(x0 + dx0, factor_x), color="gray", alpha=1)
            #plt.plot(xinterval, yinterval, ".")
    else:
        ptp_ci = yinterval_sup_ci.max() - yinterval_inf_ci.min()
        plt.ylim(max(-0.2, yinterval_inf_ci.min() - 0.2 * ptp_ci), 0.2 * ptp_ci + yinterval_sup_ci.max())
        plt.xlim(xinterval.min(), xinterval.max())
    plt.legend()


###############################################################################
# SIMULATE DATA USING PARAMETERS
###############################################################################
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


###############################################################################
# PARAMETERS
###############################################################################
ALL_PARAMETERS = "F H Q R X0 P0"

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

"""
##simulate VAR as in Enders 2004, p 268
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

def initial_parameters(type, factor_noise=0.2):
    if type == "1x3":
        params, X, Y = create_true_params_1x3()
    elif type == "1x3_true":
        params, X, Y = create_true_params_1x3_true()
    elif type == "2x2_enders":
        params, X, Y = create_true_params_enders_2x2_true()
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
# TEST ESTIMATORS
###############################################################################
def test_metaestimator(type_estimator, X, Y, params, estimates=ALL_PARAMETERS, sample_size=3000, min_iterations=5, max_iterations=20, min_improvement=0.01, population_size=10, penalty_factors={}, plot_intermediate_results=True):
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

def test_estimator(type_estimator, X, Y, params, estimates=ALL_PARAMETERS, sample_size=3000, min_iterations=5, max_iterations=20, min_improvement=0.01, population_size=10, penalty_factors={}, plot_intermediate_results=True):
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

###############################################################################
# EVALUATION METRIC
###############################################################################
def mse(Xtrue, Xest):
    #return np.mean(np.abs((Xtrue.ravel() - Xest.ravel()) ** 2))
    return np.sqrt(np.mean((Xtrue.ravel() - Xest.ravel()) ** 2))

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
def _test_set_parameters(type_estimator, X, Y, simparams, prevInErrs=[], prevOutErrs=[]):
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
                "mse": 0,#1e-1,
            },
            plot_intermediate_results=False,
        )
        
        inErr, outErr = 0, 0
        p_estimates = ALL_PARAMETERS.split()
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
        #print("o", end=" {0:.1f} ".format(np.mean(outErrs[:i+1])))
    #print()
    return np.concatenate([prevInErrs, inErrs]), np.concatenate([prevOutErrs, outErrs]), bestEstParams
    
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
            plot_error_density(results[estimates]["factors"], results[estimates]["out"], title="Estimation error of {}".format(estimates))
            plot_error_confidence_interval2(results[estimates]["factors"], results[estimates]["out"], title="Estimation error of {}".format(estimates), maxerror=0.5);plt.show()
            plot_error_confidence_interval2(results[estimates]["factors"], results[estimates]["out"], title="Estimation error of {}".format(estimates))
            plt.show()
        return results

    results = {}
    
    
    for estimates in all_estimates:
        simparams2 = [(a,b,c) for (a,b,c) in simparams if b == estimates]
        idxsimparams2 = [i for i, (a,b,c) in enumerate(simparams) if b == estimates]
        factors = np.array([c for (a,b,c) in simparams if b == estimates])
        print("Estimates", estimates)
        inErrs, outErrs, bestEstParams = _test_set_parameters(type_estimator, X, Y, simparams2)
        #'''
        # Remove outliers
        xfactors = factors.copy()
        xinErrs = inErrs.copy()
        xoutErrs = outErrs.copy()
        inErrs = inErrs[np.isfinite(outErrs)]
        factors = factors[np.isfinite(outErrs)]
        outErrs = outErrs[np.isfinite(outErrs)]
        inErrs = inErrs[np.abs(outErrs - outErrs.mean()) < 7 * outErrs.std()]
        factors = factors[np.abs(outErrs - outErrs.mean()) < 7 * outErrs.std()]
        outErrs = outErrs[np.abs(outErrs - outErrs.mean()) < 7 * outErrs.std()]
        #
        try:
            #plot_error_density(inErrs, outErrs, title="Estimation error of {}".format(estimates))
            plot_error_density(factors, outErrs, title="Estimation error of {}".format(estimates))
            #plot_error_confidence_interval(inErrs, outErrs, title="Estimation error of {}".format(estimates))
            plot_error_confidence_interval2(factors, outErrs, title="Estimation error of {}".format(estimates), maxerror=0.5);plt.show()
            plot_error_confidence_interval2(factors, outErrs, title="Estimation error of {}".format(estimates))
            plt.show()
        except Exception as e:
            print("ERROR: Plot failed!", e)
            [(plt.clf(), plt.close()) for _ in range(4)]
        results[estimates] = {
            "in": inErrs, "out": outErrs, 
            "xin": xinErrs, "xout": xoutErrs, 
            "estimator": bestEstParams, "idx": np.array(idxsimparams2),
            "factors": factors, "xfactors": xfactors,
        }
    
    if cachefile is not None:
        with open(cachefile, "wb") as f:
            pickle.dump(results, f)
    return results

