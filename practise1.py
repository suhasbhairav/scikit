import sklearn.datasets

iris_datset = sklearn.datasets.load_iris()
print(iris_datset)
X, y = iris_datset["data"], iris_datset["target"]
print(X)
print(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)

clf.score(X_test, y_test)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X, y, cv=5)
print(scores)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()/2))


