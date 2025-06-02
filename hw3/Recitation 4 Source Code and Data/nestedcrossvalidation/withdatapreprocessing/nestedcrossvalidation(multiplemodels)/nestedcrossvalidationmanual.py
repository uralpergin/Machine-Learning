from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
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

knn_performance = []
svm_performance = []

knn_overall_performance = []
svm_overall_performance = []

for train_indices, test_indices in outer_cross_validation.split(dataset, labels):
    current_training_part = dataset[train_indices]
    current_training_part_label = labels[train_indices]


    # knn_pipeline = Pipeline([('minmaxscaler', MinMaxScaler()),
    #                 ('kneighborsclassifier', KNeighborsClassifier())])
    knn_pipeline = make_pipeline(MinMaxScaler(), KNeighborsClassifier())

    knn_grid_search = GridSearchCV(knn_pipeline, param_grid=knn_parameter_grid, refit=True, cv=inner_cross_validation, scoring="f1_micro")
    knn_grid_search.fit(current_training_part, current_training_part_label)

    # svm_pipeline = Pipeline([('minmaxscaler', MinMaxScaler()),
    #                 ('svc', SVC())])

    svm_pipeline = make_pipeline(MinMaxScaler(), SVC())
    svm_grid_search = GridSearchCV(svm_pipeline, param_grid=svm_parameter_grid, refit=True, cv=inner_cross_validation, scoring="f1_micro")
    svm_grid_search.fit(current_training_part, current_training_part_label)

    current_test_part = dataset[test_indices]
    current_test_part_label = labels[test_indices]


    knn_predicted = knn_grid_search.predict(current_test_part)
    knn_overall_performance.append(f1_score(current_test_part_label, knn_predicted, average="micro"))

    svm_predicted = svm_grid_search.predict(current_test_part)
    svm_overall_performance.append(f1_score(current_test_part_label, svm_predicted, average="micro"))


print(knn_overall_performance)
print(np.mean(knn_overall_performance))

print(svm_overall_performance)
print(np.mean(svm_overall_performance))