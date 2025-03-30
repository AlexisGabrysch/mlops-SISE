from sklearn import datasets
from sklearn import tree
import joblib

iris = datasets.load_iris()
model=tree.DecisionTreeClassifier()

X = iris.data
y = iris.target
target_names = list(iris.target_names )
model.fit(X, y)

joblib.dump(model, "model.pkl")