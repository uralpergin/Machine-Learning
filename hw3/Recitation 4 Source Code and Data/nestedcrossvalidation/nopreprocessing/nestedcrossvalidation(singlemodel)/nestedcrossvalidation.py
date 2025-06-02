from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score

# to be removed
import random
random.seed(11)
np.random.seed(17)

parameter_grid = {"metric": ["cosine", "euclidean", "manhattan"],
              "n_neighbors": [2, 3, 4]
              }

knn = KNeighborsClassifier()
dataset, labels = pickle.load(open("../../../data/part2_dataset1.data", "rb"))
outer_cross_validation = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=np.random.randint(1, 1000))
inner_cross_validation = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=np.random.randint(1, 1000))
model = GridSearchCV(knn, parameter_grid, scoring="f1", cv=inner_cross_validation, verbose=True)
val = cross_val_score(model, dataset, labels, cv=outer_cross_validation, scoring="f1", verbose=True)
print(val)
