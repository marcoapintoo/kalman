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
    os.path.realpath(os.path.join(
        os.path.realpath(os.path.dirname(sys.argv[0])), "../../../bin/python/pure/"
    ))
)
sys.path.append(
    os.path.realpath(os.path.join(
        os.path.realpath(os.path.dirname(sys.argv[0])), "../../../bin/"
    ))
)
import state_space_model as ssm
ssm._test_module()

np.random.seed(42)
Y = np.random.randn(2, 100)
Y[0, :] += 10
Y[1, :] += 50
X1, Y1 = ssm.estimate_ssm("pso", estimates="F H Q R X0 P0", Y=Y, obs_dim=2, lat_dim=1, T=100)
print(Y)
print(Y1)

