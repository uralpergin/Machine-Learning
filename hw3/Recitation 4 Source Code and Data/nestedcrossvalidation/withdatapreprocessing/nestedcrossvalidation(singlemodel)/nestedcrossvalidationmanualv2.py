from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler

# to be removed
import random
random.seed(11)
np.random.seed(17)

knn_parameter_grid = {"metric": ["cosine", "euclidean", "manhattan"],
              "n_neighbors": [2, 3, 4]
              }


dataset, labels = pickle.load(open("../../../data/part2_dataset2.data", "rb"))

outer_cross_validation = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=np.random.randint(1, 1000))
inner_cross_validation = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=np.random.randint(1, 1000))

knn_parameter_performance = dict()

knn_overall_performance = []

for train_indices, test_indices in outer_cross_validation.split(dataset, labels):
    current_training_part = dataset[train_indices]
    current_training_part_label = labels[train_indices]

    current_test_part = dataset[test_indices]
    current_test_part_label = labels[test_indices]


    knn_performance = dict()

    for inner_train_indices, inner_test_indices in inner_cross_validation.split(current_training_part, current_training_part_label):

        inner_training_dataset = current_training_part[inner_train_indices]
        inner_training_label = current_training_part_label[inner_train_indices]

        inner_test_dataset = current_training_part[inner_test_indices]
        inner_test_label = current_training_part_label[inner_test_indices]

        inner_scaler = MinMaxScaler()
        inner_scaler.fit(inner_training_dataset)
        scaled_inner_training_dataset = inner_scaler.transform(inner_training_dataset)

        scaled_inner_test_dataset = inner_scaler.transform(inner_test_dataset)

        for n_neighbor in knn_parameter_grid["n_neighbors"]:
            for metric in knn_parameter_grid["metric"]:

                knn = KNeighborsClassifier(metric=metric, n_neighbors=n_neighbor)


                knn.fit(scaled_inner_training_dataset, inner_training_label)

                predicted = knn.predict(scaled_inner_test_dataset)
                if (metric, n_neighbor) not in knn_performance:
                    knn_performance[(metric, n_neighbor)] = []
                knn_performance[(metric, n_neighbor)].append(f1_score(inner_test_label, predicted, average="micro"))


    best_parameter_knn = None
    best_score_knn = -float('inf')

    for param_config in knn_performance:
        v = np.mean(knn_performance[param_config])
        if v > best_score_knn:
            best_score_knn = v
            best_parameter_knn = param_config

    print(best_parameter_knn)
    outer_scaler = MinMaxScaler()
    outer_scaler.fit(current_training_part)

    knn_with_best_param = KNeighborsClassifier(n_neighbors=best_parameter_knn[1],metric=best_parameter_knn[0])


    knn_with_best_param.fit(outer_scaler.transform(current_training_part), current_training_part_label)


    knn_predicted = knn_with_best_param.predict(outer_scaler.transform(current_test_part))
    knn_overall_performance.append(f1_score(current_test_part_label, knn_predicted, average="micro"))


print(knn_overall_performance)
print(np.mean(knn_overall_performance))

