import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

my_data = pd.read_csv("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/drug200.csv", delimiter=",")

X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
X[0:5]

from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1])


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3])

X[0:5]

y = my_data["Drug"]
y[0:5]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)
print('Train set:', X_train.shape,  y_train.shape)
print('Test set:', X_test.shape,  y_test.shape)

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
drugTree

drugTree.fit(X_train, y_train)

predTree = drugTree.predict(X_test)
predTree[:5]
y_test[:5]

from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predTree))

from six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
from IPython.display import Image

featureNames = my_data.columns[0:5]
targetNames = my_data["Drug"].unique().tolist()

dot_data = tree.export_graphviz(drugTree, out_file=None,
                                feature_names=featureNames,
                                class_names=targetNames)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
