from sklearn.neighbors import KNeighborsClassifier
import sklearn
import sklearn.datasets
import numpy as np
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

print(cross_val_score(gridsearch_cv, dataset, labels, cv=cross_validation, verbose=True, scoring="accuracy"))

