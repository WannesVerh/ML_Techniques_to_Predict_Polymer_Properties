from pyexpat import model
from matplotlib.lines import lineMarkers
import numpy as np
import matplotlib.pyplot as plt
import deepchem as dc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.gaussian_process.kernels import RBF

X = np.linspace(start=0, stop=10, num=1_000).reshape(-1, 1)
y = np.squeeze(X * np.sin(X))
plt.plot(X,y,label='f(x)=xsin(x)',linestyle='dotted')
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("true data points")
plt.show()


###noise free target

range = np.random.RandomState(1) #generates numbers arround 1
training_indices = range.choice(np.arange(y.size), size=7, replace=False)
X_train, y_train = X[training_indices], y[training_indices]#generate date that diviates from the sinus function to train model

trainData = dc.data.NumpyDataset(X_train,y_train)
testData = dc.data.NumpyDataset(X)

#splitter =  dc.splits.RandomSplitter()
#trainData, validData, testData = splitter.train_valid_test_split(dataset=dataset, frac_train=0.7, frac_valid=0.2, frac_test=0.1)

kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
model = dc.models.SklearnModel(GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9))

print("fitting model")
model.fit(trainData)
print("model is fitted")
mean_prediction = model.predict(testData)
std_prediction = mean_absolute_error(y,mean_prediction, multioutput='raw_values')


##plot the data
plt.plot(X, y, label="f(x)=x\sin(x)", linestyle="dotted")
plt.scatter(X_train, y_train, label="Observations")
plt.plot(X, mean_prediction, label="Mean prediction")
plt.fill_between(
    X.ravel(),
    mean_prediction - 1.96 * std_prediction,
    mean_prediction + 1.96 * std_prediction,
    alpha=0.5,
    label=r"95% confidence interval",
)
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.title("Gaussian process regression on noise-free dataset")
plt.show()





""""
from sklearn.datasets import fetch_openml
co2 = fetch_openml(data_id=41187, as_frame=True)
print(co2.frame.head())"""


"""
import numpy as np

X = np.linspace(start=0, stop=10, num=1_000).reshape(-1, 1)
y = np.squeeze(X * np.sin(X))
import matplotlib.pyplot as plt

plt.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.title("True generative process")
rng = np.random.RandomState(1)
training_indices = rng.choice(np.arange(y.size), size=6, replace=False)
X_train, y_train = X[training_indices], y[training_indices]
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
print(X_train,y_train)
gaussian_process.fit(X_train, y_train)
gaussian_process.kernel_
mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

plt.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
plt.scatter(X_train, y_train, label="Observations")
plt.plot(X, mean_prediction, label="Mean prediction")
plt.fill_between(
    X.ravel(),
    mean_prediction - 1.96 * std_prediction,
    mean_prediction + 1.96 * std_prediction,
    alpha=0.5,
    label=r"95% confidence interval",
)
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.title("Gaussian process regression on noise-free dataset")
plt.show()"""