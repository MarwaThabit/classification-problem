from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

def random_forest_classifier(X_train, y_train):
    model = RandomForestClassifier(n_estimators=120, max_depth=None,random_state=1)
    model.fit(X_train, y_train)
    return model

C=2
def svm_classifier(X_train, y_train):
    model = SVC(C=C )
    model.fit(X_train, y_train)
    return model

def knn_classifier(X_train, y_train):
    model = KNeighborsClassifier(n_neighbors=7, weights='distance')
    model.fit(X_train, y_train)
    return model


def svm_linear_classifier(X_train, y_train):
    model = SVC(kernel='linear',C=C)
    model.fit(X_train, y_train)
    return model

def svm_polynomial_classifier(X_train, y_train, degree=3):
    model = SVC(kernel='poly', degree=degree,C=C)
    model.fit(X_train, y_train)
    return model

def svm_rbf_classifier(X_train, y_train):
    model = SVC(kernel='rbf', gamma='scale',C=2)
    model.fit(X_train, y_train)
    return model


def decision_tree_classifier(X_train, y_train):
    model = DecisionTreeClassifier(random_state=1,max_depth=3)
    model.fit(X_train, y_train)
    return model

def logistic_regression_classifier(X_train, y_train):
    model = LogisticRegression(random_state=1,C=2)
    model.fit(X_train, y_train)
    return model