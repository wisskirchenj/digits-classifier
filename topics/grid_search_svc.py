from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
scaler = StandardScaler()
X = scaler.fit_transform(X)

kernel = ['linear', 'rbf', 'poly']
gamma = [0.1, 1, 10, 100]
C = [0.1, 1, 10, 100, 1000]

param_grid = {'kernel': kernel, 'gamma': gamma, 'C': C}
grid = GridSearchCV(SVC(), param_grid)
grid.fit(X, y)
print(grid.best_params_)
print(grid.best_estimator_)
print(grid.best_score_)
