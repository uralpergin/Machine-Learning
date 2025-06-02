from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import sklearn.datasets


dataset, labels = sklearn.datasets.load_wine(return_X_y=True)

new_dataset = []
new_labels = []
positive_count = 0
negative_count = 0
for i in range(len(dataset)):
    if positive_count < 30 and labels[i] == 1:
        new_dataset.append(dataset[i])
        new_labels.append(labels[i])
        positive_count += 1

    if negative_count < 30 and labels[i] == 0:
        new_dataset.append(dataset[i])
        new_labels.append(labels[i])
        negative_count += 1


dataset = new_dataset
labels = new_labels

dt = DecisionTreeClassifier()
dt.fit(dataset, labels)

tree = RandomForestClassifier()
tree.fit(dataset, labels)

print("Feature importance : ", tree.feature_importances_)

for i, a in enumerate(tree.feature_importances_):
    print(i, a)

plot_tree(tree.estimators_[0], rounded=True, precision=10)
plt.show()

plot_tree(tree.estimators_[1], rounded=True, precision=10)
plt.show()
# parameter importances...
# show tree