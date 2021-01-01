import sklearn.datasets
iris_datset = sklearn.datasets.load_iris()

X, y = iris_datset["data"], iris_datset["target"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80)

from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=2)
clf.fit(X_train, y_train)

clf.score(X_test, y_test)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X, y, cv=6)
print(scores)

print("Accuracy: %0.2f (+/-) %.02f" % (scores.mean(), (scores.std()/2)))
