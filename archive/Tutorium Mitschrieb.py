import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

iris = datasets.load_iris()
X = iris.data[:, :2]
y = np.array(iris.target, dtype=int)
kernel = 1.0 * RBF([1.0])
gpc_rbf_isotropic = GaussianProcessClassifier(kernel=kernel)
fitted_gpc_rbf_isotropic = gpc_rbf_isotropic.fit(X, y)
#...(http://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpc_iris.html)