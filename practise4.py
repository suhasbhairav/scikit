import sklearn.datasets
import numpy as np
iris_dataset = sklearn.datasets.load_iris()
print(iris_dataset.keys())

X, y = iris_dataset['data'], iris_dataset["target"]
print(X.shape, y.shape)

print(np.unique(y))

boston_dataset = sklearn.datasets.load_boston()
print(boston_dataset.keys())

X, y = boston_dataset["data"], boston_dataset["target"]
print(X.shape, y.shape)
print(np.unique(y))

breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
print(breast_cancer_dataset.keys())

X, y = breast_cancer_dataset["data"], breast_cancer_dataset["target"]
print(X.shape, y.shape)

print(np.unique(y))

diabetes_dataset = sklearn.datasets.load_diabetes()
print(diabetes_dataset.keys())

X, y = diabetes_dataset["data"], diabetes_dataset["target"]
print(X.shape, y.shape)

print(np.unique(y))

load_digits_dataset = sklearn.datasets.load_digits()
print(load_digits_dataset.keys())

X, y = load_digits_dataset["data"], load_digits_dataset["target"]
print(X.shape, y.shape)

print(np.unique(y))