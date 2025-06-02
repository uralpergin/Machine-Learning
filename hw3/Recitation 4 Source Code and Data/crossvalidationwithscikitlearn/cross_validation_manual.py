from sklearn.neighbors import KNeighborsClassifier
import sklearn.datasets
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
import random

np.random.seed(11)
random.seed(17)

dataset, label = sklearn.datasets.load_wine(return_X_y=True)
print(dataset.shape)

print(dataset)
kfold = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=11)

parameter_grid = {"metric": ["cosine", "euclidean", "manhattan"],
                  "n_neighbors": [2, 3]
              }

best_test_scores = []
PERFORMANCE_LOG = dict()

for train_indices, test_indices in kfold.split(dataset, label):
    current_train = dataset[train_indices]
    current_train_label = label[train_indices]

    current_test = dataset[test_indices]
    current_test_labels = label[test_indices]

    for metric in parameter_grid["metric"]:
        if metric not in PERFORMANCE_LOG:
            PERFORMANCE_LOG[metric] = dict()
        for k in parameter_grid["n_neighbors"]:
            if k not in PERFORMANCE_LOG[metric]:
                PERFORMANCE_LOG[metric][k] = []
            # if it were a randomized algorithm, we should train it multiple times
            # to obtain an average performance score with a particular partitioning and hyperparameter configuration
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
            knn.fit(current_train, current_train_label)

            predicted = knn.predict(current_test)
            accuracy = accuracy_score(current_test_labels, predicted)
            PERFORMANCE_LOG[metric][k].append(accuracy)

    fold_best_score = -float('inf')
    average = 0
    for metric in PERFORMANCE_LOG:
        for k in PERFORMANCE_LOG[metric]:
            performance = PERFORMANCE_LOG[metric][k][-1]
            average += performance
            if performance > fold_best_score:
                fold_best_score = performance
    best_test_scores.append((fold_best_score))



best_config = None
best_score = -float('inf')
for metric in PERFORMANCE_LOG:
    for k in PERFORMANCE_LOG[metric]:
        test_scores = PERFORMANCE_LOG[metric][k]
        mean_value = np.mean(test_scores)
        std_value = np.std(test_scores)
        print(f"metric {metric} - k {k} : {mean_value:.3f}")
        if mean_value > best_score:
            best_score = mean_value
            best_config = (metric, k)


print("Best hyperparameter configuration : ", best_config)
final_mode = KNeighborsClassifier(n_neighbors=best_config[1], metric=best_config[0])
knn.fit(dataset, label)
predicted = knn.predict(dataset)
accuracy = accuracy_score(label, predicted)
print(accuracy)
print("Best parameter scores : ", PERFORMANCE_LOG[best_config[0]][best_config[1]])

print(PERFORMANCE_LOG)
print("Best fold scores : ", best_test_scores)
