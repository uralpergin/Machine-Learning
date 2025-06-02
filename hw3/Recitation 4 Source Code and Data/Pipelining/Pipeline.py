from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
import sklearn.datasets

dataset, labels = sklearn.datasets.load_wine(return_X_y=True)


X_train, X_test, y_train, y_test = train_test_split(dataset, labels, shuffle=True, test_size=0.3, random_state=42, stratify=labels)

# pipeline = Pipeline([('minmaxscaler', MinMaxScaler()),
#                 ('kneighborsclassifier', KNeighborsClassifier())])
pipeline = make_pipeline(MinMaxScaler(), KNeighborsClassifier())
pipeline.fit(X_train, y_train)

predicted = pipeline.predict(X_test)
print("Accuracy : ", accuracy_score(y_test, predicted))

print(pipeline)

print(pipeline.steps)
processed = pipeline.steps[0][1].transform(X_test)
predicted = pipeline.steps[1][1].predict(processed)

print("Accuracy : ", accuracy_score(y_test, predicted))











