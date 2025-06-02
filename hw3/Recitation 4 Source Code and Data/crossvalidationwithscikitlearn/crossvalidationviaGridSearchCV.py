from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np
import sklearn.datasets
import random
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

np.random.seed(11)
random.seed(17)

parameter_grid = {"metric": ["cosine", "euclidean", "manhattan"],
                  "n_neighbors": [2, 3]
              }

knn = KNeighborsClassifier()
dataset, labels = sklearn.datasets.load_wine(return_X_y=True)

cross_validation = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=11)


gridsearch_cv = GridSearchCV(knn, parameter_grid, scoring="accuracy", cv=cross_validation, verbose=True, refit=True)
gridsearch_cv.fit(dataset, labels)

print(gridsearch_cv.cv_results_)
print(gridsearch_cv.best_params_)

predicted = gridsearch_cv.best_estimator_.predict(dataset)
print(accuracy_score(labels, predicted)*100)

