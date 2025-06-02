from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
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

dataset, labels = pickle.load(open("../../../data/part2_dataset1.data", "rb"))

outer_cross_validation = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=np.random.randint(1, 1000))
inner_cross_validation = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=np.random.randint(1, 1000))

knn_parameter_performance = dict()

knn_overall_performance = []
svm_overall_performance = []

for train_indices, test_indices in outer_cross_validation.split(dataset, labels):
    current_training_part = dataset[train_indices]
    current_training_part_label = labels[train_indices]

    current_test_part = dataset[test_indices]
    current_test_part_label = labels[test_indices]

    knn_performance = dict()
    svm_performance = dict()

    for inner_train_indices, inner_test_indices in inner_cross_validation.split(current_training_part, current_training_part_label):

        inner_training_dataset = current_training_part[inner_train_indices]
        inner_training_label = current_training_part_label[inner_train_indices]

        inner_test_dataset = current_training_part[inner_test_indices]
        inner_test_label = current_training_part_label[inner_test_indices]


        for n_neighbor in knn_parameter_grid["n_neighbors"]:
            for metric in knn_parameter_grid["metric"]:

                knn = KNeighborsClassifier(metric=metric, n_neighbors=n_neighbor)
                knn.fit(inner_training_dataset, inner_training_label)

                predicted = knn.predict(inner_test_dataset)
                if (metric, n_neighbor) not in knn_performance:
                    knn_performance[(metric, n_neighbor)] = []
                knn_performance[(metric, n_neighbor)].append(f1_score(inner_test_label, predicted))


        for C in svm_parameter_grid["C"]:
            for kernel in svm_parameter_grid["kernel"]:

                svm = SVC(C=C, kernel=kernel)
                svm.fit(inner_training_dataset, inner_training_label)

                predicted = svm.predict(inner_test_dataset)
                if (C, kernel) not in svm_performance:
                    svm_performance[(C, kernel)] = []
                svm_performance[(C, kernel)].append(f1_score(inner_test_label, predicted))


    best_parameter_knn = None
    best_score_knn = -float('inf')

    for param_config in knn_performance:
        v = np.mean(knn_performance[param_config])
        if v > best_score_knn:
            best_score_knn = v
            best_parameter_knn = param_config

    print(best_parameter_knn)

    best_parameter_svm = None
    best_score_svm = -float('inf')

    for param_config in svm_performance:
        v = np.mean(svm_performance[param_config])
        if v > best_score_svm:
            best_score_svm = v
            best_parameter_svm = param_config

    print(best_parameter_svm)





    knn_with_best_param = KNeighborsClassifier(n_neighbors=best_parameter_knn[1],metric=best_parameter_knn[0])
    knn_with_best_param.fit(current_training_part, current_training_part_label)

    svm_with_best_param = SVC(C=best_parameter_svm[0], kernel=best_parameter_svm[1])
    svm_with_best_param.fit(current_training_part, current_training_part_label)




    knn_predicted = knn_with_best_param.predict(current_test_part)
    knn_overall_performance.append(f1_score(current_test_part_label, knn_predicted))

    svm_predicted = svm_with_best_param.predict(current_test_part)
    svm_overall_performance.append(f1_score(current_test_part_label, svm_predicted))


print(knn_overall_performance)
print(np.mean(knn_overall_performance))

print(svm_overall_performance)
print(np.mean(svm_overall_performance))

