from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
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

knn_parameter_grid = {"kneighborsclassifier__metric": ["cosine", "euclidean", "manhattan"],
                          "kneighborsclassifier__n_neighbors": [2, 3, 4]
                          }

svm_parameter_grid = {"svc__C": [0.1, 0.5],
              "svc__kernel": ["poly", "rbf"]
}


dataset, labels = pickle.load(open("../../../data/part2_dataset2.data", "rb"))

outer_cross_validation = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=np.random.randint(1, 1000))
inner_cross_validation = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=np.random.randint(1, 1000))
# knn_pipeline = Pipeline([('minmaxscaler', MinMaxScaler()),
#                 ('kneighborsclassifier', KNeighborsClassifier())])
knn_pipeline = make_pipeline(MinMaxScaler(), KNeighborsClassifier())
knn_grid = GridSearchCV(knn_pipeline, knn_parameter_grid, scoring="f1_micro", cv=inner_cross_validation, verbose=True)
knn_val = cross_val_score(knn_grid, dataset, labels, cv=outer_cross_validation, verbose=True)
print(np.mean(knn_val))
# svm_pipeline = Pipeline([('minmaxscaler', MinMaxScaler()),
#                 ('svc', SVC())])
svm_pipeline = make_pipeline(MinMaxScaler(), SVC())
svm_grid = GridSearchCV(svm_pipeline, svm_parameter_grid, scoring="f1_micro", cv=inner_cross_validation, verbose=True)
svm_val = cross_val_score(svm_grid, dataset, labels, cv=outer_cross_validation, scoring="f1_micro", verbose=True)
print((svm_val))



