from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
scaler = StandardScaler()
X = scaler.fit_transform(X)

kernel = ['linear', 'rbf', 'poly']
C = loguniform(0.01, 1)
gamma = loguniform(0.01, 1)

param_distributions = {'kernel': kernel, 'C': C, 'gamma': gamma}
random_search = RandomizedSearchCV(SVC(), param_distributions, random_state=42, scoring='top_k_accuracy')
random_search.fit(X, y)
print(random_search.best_params_)

