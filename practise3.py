import sklearn.datasets

iris_dataset = sklearn.datasets.load_iris()
X, y = iris_dataset["data"], iris_dataset["target"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=2)
clf.fit(X_train, y_train)

clf.score(X_test, y_test)

from sklearn.model_selection import  cross_val_score
scores = cross_val_score(clf, X, y, cv=5)
print("Accuracy %0.2f (+/-%0.2f)" % (scores.mean(), scores.std()/2))


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=4))

pipeline.fit(X_train, y_train)
print(pipeline.predict_proba(X_test))

