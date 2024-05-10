from scipy.stats import loguniform
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

random_state = 42

X, y = load_breast_cancer(return_X_y=True)
scaler = StandardScaler()
X = scaler.fit_transform(X)

kernel = ['linear', 'rbf', 'poly']
param_grid = {"kernel": kernel, "gamma": loguniform(0.01, 1), "C": loguniform(0.01, 1)}
random_search = RandomizedSearchCV(SVC(), param_grid, random_state=random_state)
random_search.fit(X, y)
print(round(random_search.cv_results_['mean_test_score'][3], 2))
