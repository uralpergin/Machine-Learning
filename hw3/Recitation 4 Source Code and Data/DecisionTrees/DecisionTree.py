from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import sklearn.datasets

dataset, labels = sklearn.datasets.load_wine(return_X_y=True)
print(labels)
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

tree = DecisionTreeClassifier()
tree.fit(dataset, labels)

print("Feature importance : ", tree.feature_importances_)
plot_tree(tree, rounded=True, precision=10)
for i, a in enumerate(tree.feature_importances_):
    print(i, a)
plt.show()
# parameter importances...
# show tree