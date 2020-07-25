import pickle
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


MODEL_MAPPING = {'random_forest': RandomForestClassifier(n_estimators=10,
                                                         max_depth=3, random_state=42),
                 'logistic': LogisticRegression(C=10, random_state=42),
                 'svm': SVC(C=10,gamma='auto'),
                 'knn': KNeighborsClassifier(),
                 'decision_tree': DecisionTreeClassifier()}

PARAMETER_MAPPING = {'random_forest': {'n_estimators': list(range(10, 20)), 'max_depth': [3]},
                     'logistic': {'penalty': ('l1', 'l2'), 'C': [5, 10]},
                     'svm': {'kernel': ('linear', 'rbf'), 'C': [5, 10]},
                     'knn': {},
                     'decision_tree': {'max_depth': [3], 'min_samples_leaf': [2, 3, 4, 5]},
                     }


def train_model(model_name, X_train, y_train):
    if model_name in MODEL_MAPPING.keys():
        model = MODEL_MAPPING[model_name]
        parameters = PARAMETER_MAPPING[model_name]
        clf = GridSearchCV(model, parameters, cv=5)
        clf.fit(X_train, y_train)
        return clf
    else:
        print(
            f"please pass the model name one of these : {list(MODEL_MAPPING.keys())}")


def save_model(filepath, model):
    joblib.dump(model, filepath)


def load_model(filepath):
    model = joblib.load(filepath)
    return model