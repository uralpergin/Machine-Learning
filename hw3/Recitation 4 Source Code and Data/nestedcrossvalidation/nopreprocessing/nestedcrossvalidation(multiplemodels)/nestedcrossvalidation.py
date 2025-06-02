from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score

# to be removed
import random
random.seed(11)
np.random.seed(17)

knn_parameter_grid = {"metric": ["cosine", "euclidean", "manhattan"],
              "n_neighbors": [2, 3, 4]
              }

svm_parameter_grid = {"C": [0.1, 0.5],
              "kernel": ["poly", "rbf"]
}

knn = KNeighborsClassifier()
svm = SVC()
dataset, labels = pickle.load(open("../../../data/part2_dataset1.data", "rb"))

outer_cross_validation = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=np.random.randint(1, 1000))
inner_cross_validation = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=np.random.randint(1, 1000))


knn_grid = GridSearchCV(knn, knn_parameter_grid, scoring="f1", cv=inner_cross_validation, verbose=True)
knn_val = cross_val_score(knn_grid, dataset, labels, cv=outer_cross_validation, verbose=True)
print(knn_val)

svm_grid = GridSearchCV(svm, svm_parameter_grid, scoring="f1", cv=inner_cross_validation, verbose=True)
svm_val = cross_val_score(svm_grid, dataset, labels, cv=outer_cross_validation, scoring="f1", verbose=True)
print(svm_val)



