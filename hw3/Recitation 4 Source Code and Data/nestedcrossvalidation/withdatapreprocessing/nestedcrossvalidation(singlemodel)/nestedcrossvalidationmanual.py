from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

# to be removed
import random
random.seed(11)
np.random.seed(17)

dataset, labels = pickle.load(open("../../../data/part2_dataset2.data", "rb"))

outer_cross_validation = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=np.random.randint(1, 1000))
inner_cross_validation = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=np.random.randint(1, 1000))

knn_overall_performance = []
for train_indices, test_indices in outer_cross_validation.split(dataset, labels):
    current_training_part = dataset[train_indices]
    current_training_part_label = labels[train_indices]


    # pipeline = Pipeline([('minmaxscaler', MinMaxScaler()),
    #                 ('kneighborsclassifier', KNeighborsClassifier())])
    pipeline = make_pipeline(MinMaxScaler(), KNeighborsClassifier())

    knn_parameter_grid = {"kneighborsclassifier__metric": ["cosine", "euclidean", "manhattan"],
                          "kneighborsclassifier__n_neighbors": [2, 3, 4]
                          }
    knn_grid_search = GridSearchCV(pipeline, param_grid=knn_parameter_grid, refit=True, cv=inner_cross_validation, scoring="f1_micro")
    knn_grid_search.fit(current_training_part, current_training_part_label)

    current_test_part = dataset[test_indices]
    current_test_part_label = labels[test_indices]


    predicted = knn_grid_search.best_estimator_.predict(current_test_part)
    knn_overall_performance.append(f1_score(current_test_part_label, predicted, average="micro"))

print(knn_overall_performance)
print(np.mean(knn_overall_performance))