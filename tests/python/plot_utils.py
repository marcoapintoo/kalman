import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
#plt.style.use("ggplot")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (7.5, 5)

#### plot_likelihood_record => plot_likelihood_record
def plot_likelihood_record(records):
    plt.figure(figsize=(15, 3))
    plt.plot(records, "-", color="red")
    plt.plot(records, "o", color="black")
    plt.xlabel("Iterations")
    plt.xlabel("Loglikelihood")
    plt.title("Loglikelihood improvements")

#### plot_data => plot_smoothed_predicted_data
def plot_smoothed_predicted_data(Xs, Ys, Yp=None, Xp=None, X=None, Y=None):
    nrows = max(Xs.shape[0], Ys.shape[0])
    plt.figure(figsize=(15, 5 * nrows))
    labels = ["Kalman smoothed signal"]
    predicted = Yp is not None and Xp is not None
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

#### plot_simple_data => plot_system_signals
def plot_system_signals(Xs, Ys):
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

def get_Xs(smoother):
    return smoother[6]
    
def get_ys(smoother):
    return smoother[8]

def get_loglikelihood_record(smoother):
    return smoother[9]

#### plot_smoothers => plot_smoother_signals
def plot_smoother_signals(smoothers):
    Xs = get_Xs(smoothers[0][1])
    Ys = get_Ys(smoothers[0][1])
    nrows = max(Xs.shape[0], Ys.shape[0])
    plt.figure(figsize=(15, 5 * nrows))
    for k in range(Xs.shape[0]):
        plt.subplot(nrows, 2, 2 * k + 2)
        plt.title("System input (dim. {0})".format(k + 1))
        labels = []
        for name, smoother in smoothers:
            labels.append("{0}".format(name))
            plt.plot(get_Xs(smoother)[k, :].ravel(), linestyle="-")
        plt.legend(labels)
    for k in range(Ys.shape[0]):
        plt.subplot(nrows, 2, 2 * k + 1)
        plt.title("System output (dim. {0})".format(k + 1))
        labels = []
        for name, smoother in smoothers:
            labels.append("{0}".format(name))
            plt.plot(get_Ys(smoother)[k, :].ravel(), linestyle="-")
        plt.legend(labels)
    plt.tight_layout()
    plt.show()


def plot_error_density_classic(inErrs, outErrs, title="Errors"):
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

    #sns.kdeplot(inErrs, bw="scott", kernel="triw", shade=True, ax=axHistx)
    #sns.kdeplot(outErrs, bw='scott', kernel="triw", shade=True, ax=axHisty, vertical=True)
    sns.distplot(inErrs, bins=100, kde_kws=dict(bw="scott", kernel="triw", shade=True), ax=axHistx)
    sns.distplot(outErrs, bins=100, kde_kws=dict(bw='scott', kernel="triw", shade=True), ax=axHisty, vertical=True)

    qntts, bins = np.histogram(outErrs, bins=100)
    #plt.hist(outErrs, bins=100, orientation='vertical')
    #bins_x = bins[:-1] + 0.5 * (bins[1] - bins[0])
    #axHisty.plot(bins_x, qntts, "-")
    #axHisty.fill_between(bins_x, 0, qntts, alpha=0.5)
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
    #plt.hist(inErrs, bins=100, orientation='horizontal')
    #bins_x = bins[:-1] + 0.5 * (bins[1] - bins[0])
    #axHistx.plot(qntts, bins_x, "-")
    #axHistx.fill_between(bins_x, 0, qntts, alpha=0.5)
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

#plot_errors(inErrs, outErrs, title="Estimation error of {}".format("F"))

def plot_error_confidence_interval(inErrs, outErrs, intervals = 50, title="Estimation error of X"):
    inErrs, outErrs = np.round(inErrs, 5), np.round(outErrs, 5)
    plt.figure(figsize=(15, 10))
    plt.title(title)
    def bootstrap(y, measure):
        sample_size = max(5, y.shape[0] // 10)
        ys = []
        for i in range(10):
            yb = np.random.choice(y, sample_size)
            ys.append(measure(yb))
        return np.mean(ys)

    def bootstrap2(y, measure):
        sample_size = max(5, y.shape[0] // 10)
        ys = []
        for i in range(100):
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
    #delta_xinterval = xinterval[1] - xinterval[0]
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

def show():
    plt.show()

def close():
    [(plt.clf(), plt.close()) for _ in range(10)]



