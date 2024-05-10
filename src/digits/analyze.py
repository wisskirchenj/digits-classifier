from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from digits.load_data import DataSet


class Analyzer:
    def __init__(self, data_set: DataSet, seed=40):
        self.data_set = data_set
        self.classifiers = [
            (KNeighborsClassifier(), 'K-nearest neighbours algorithm', {
                'n_neighbors': [3, 4],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'brute']}
             ),
            (RandomForestClassifier(random_state=seed), 'Random forest algorithm', {
                'n_estimators': [300, 588],
                'max_features': ['sqrt', 'log2'],
                'class_weight': ['balanced', 'balanced_subsample']}
             )]
        self.results = {}

    @staticmethod
    def fit_predict_eval(model_with_params, features_train, features_test, target_train, target_test):
        grid = GridSearchCV(model_with_params[0], model_with_params[2], scoring='accuracy', n_jobs=-1)
        grid.fit(features_train, target_train)
        target_predict = grid.predict(features_test)
        score = accuracy_score(target_test, target_predict)
        return grid.best_estimator_, score

    def analyze(self) -> dict:
        for classifier in self.classifiers:
            self.results[classifier[1]] = self.fit_predict_eval(
                classifier,
                self.data_set.x_train, self.data_set.x_test,
                self.data_set.y_train, self.data_set.y_test)
        return self.results
