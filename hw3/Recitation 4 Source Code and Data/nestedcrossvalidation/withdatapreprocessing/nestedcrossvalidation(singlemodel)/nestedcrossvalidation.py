from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

# to be removed
import random
random.seed(11)
np.random.seed(17)

parameter_grid = {"kneighborsclassifier__metric": ["cosine", "euclidean", "manhattan"],
                          "kneighborsclassifier__n_neighbors": [2, 3, 4]
                 }

knn = KNeighborsClassifier()
dataset, labels = pickle.load(open("../../../data/part2_dataset2.data", "rb"))
outer_cross_validation = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=np.random.randint(1, 1000))
inner_cross_validation = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=np.random.randint(1, 1000))
# pipeline = Pipeline([('minmaxscaler', MinMaxScaler()),
#                 ('kneighborsclassifier', KNeighborsClassifier())])
pipeline = make_pipeline(MinMaxScaler(), KNeighborsClassifier())
print(pipeline)
grid_search = GridSearchCV(pipeline, parameter_grid, scoring="f1_micro", cv=inner_cross_validation, verbose=True)
val = cross_val_score(grid_search, dataset, labels, cv=outer_cross_validation, scoring="f1_micro", verbose=True)
print(val)
