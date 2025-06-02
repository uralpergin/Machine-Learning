from sklearn.svm import LinearSVC, SVC
import pickle
import numpy as np

from sklearn.metrics import accuracy_score, f1_score
from SVMs.boundary_display import model_display_boundary

import sklearn.datasets


input_data = np.array([[0.4, 0.4], # first data instance [0.4, 0.4]
                       [-0.4, 0.4], # second data instance [-0.4, 0.4]
                       [0.4, 0.6],
                       [-0.3,0.2],
                       [0, 0.7],
                       [-0.5, -0.6]],
                      dtype=np.float32)
input_data_label = np.array([1, -1, 1, -1, 1, -1], dtype=np.int32)

# a linearly non-separable dataset and labels
nonlinearly_separable_inpur_data = np.array([[0.2, 0.1],
                       [0.1, 1.3],
                       [0.1, 0.6],
                       [-1.0,0.6],
                       [0.3, 0.7],
                       [-0.1, -0.8]],
                      dtype=np.float32)

nonlinearly_separable_input_data_label = np.array([1, -1, 1, -1, 1, -1], dtype=np.int32)


iris = sklearn.datasets.load_iris()
dataset, labels = input_data, input_data_label
# dataset, labels = nonlinearly_separable_inpur_data, nonlinearly_separable_input_data_label

print(dataset)

# poly, ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
svm = SVC(C=1, kernel="rbf", degree=5)
svm.fit(dataset, labels)

predicted = svm.predict(dataset)
print("Accuracy : ", accuracy_score(labels, predicted))
# print(svm.support_vectors_)


model_display_boundary(dataset, svm, labels)