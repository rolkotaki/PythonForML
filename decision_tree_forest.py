from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier
from sklearn import datasets, tree
from sklearn.feature_selection import SelectFromModel
import pydotplus
from IPython.display import Image
import numpy as np
import matplotlib.pyplot as plt


# Training a Decision Tree Classifier

iris = datasets.load_iris()
features = iris.data
target = iris.target
# Create decision tree classifier object
decisiontree = DecisionTreeClassifier(random_state=0)
# Train model
model = decisiontree.fit(features, target)
# Make new observation
observation = [[5,  4,  3,  2]]
# Predict observation's class
print(model.predict(observation))
# View predicted class probabilities for the three classes
print(model.predict_proba(observation))

# If we want to use a different impurity measurement we can use the criterion parameter:
# Create decision tree classifier object using entropy
decisiontree_entropy = DecisionTreeClassifier(criterion='entropy', random_state=0)
model_entropy = decisiontree_entropy.fit(features, target)


# Training a Decision Tree Regressor

boston = datasets.load_boston()
features = boston.data[:, 0:2]  # Load data with only two features
target = boston.target
# Create decision tree classifier object
decisiontree = DecisionTreeRegressor(random_state=0)
# Train model
model = decisiontree.fit(features, target)
# Make new observation
observation = [[0.02, 16]]
# Predict observation's value
print(model.predict(observation))

# We can use the criterion parameter to select the desired measurement of split quality. For example, we can construct
# a tree whose splits reduce mean absolute error (MAE):
decisiontree_mae = DecisionTreeRegressor(criterion="mae", random_state=0)
model_mae = decisiontree_mae.fit(features, target)


# Visualizing a Decision Tree Model

iris = datasets.load_iris()
features = iris.data
target = iris.target
# Create decision tree classifier object
decisiontree = DecisionTreeClassifier(random_state=0)
# Train model
model = decisiontree.fit(features, target)
# Create DOT data
dot_data = tree.export_graphviz(decisiontree,
                                out_file=None,
                                feature_names=iris.feature_names,
                                class_names=iris.target_names)
# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)
# Show graph
# Image(graph.create_png())
# Create PDF
# graph.write_pdf("iris.pdf")


# Training a Random Forest Classifier

iris = datasets.load_iris()
features = iris.data
target = iris.target
# Create random forest classifier object
randomforest = RandomForestClassifier(random_state=0, n_jobs=-1)
# Train model
model = randomforest.fit(features, target)
# Make new observation
observation = [[5,  4,  3,  2]]
# Predict observation's class
print(model.predict(observation))

# Create random forest classifier object using entropy
randomforest_entropy = RandomForestClassifier(criterion="entropy", random_state=0)
# Train model
model_entropy = randomforest_entropy.fit(features, target)

# A common problem with decision trees is that they tend to fit the training data too closely (i.e., overfitting).
# This has motivated the widespread use of an ensemble learning method called random forest. In a random forest, many
# decision trees are trained, but each tree only receives a bootstrapped sample of observations and each node only
# considers a subset of features when determining the best split. This forest of randomized decision trees votes to
# determine the predicted class.

# Some important RandomForestClassifier parameters:
# max_features: determines the maximum number of features to be considered at each node and takes a number of arguments
#   integers (number of features)
#   floats (percentage of features)
#   sqrt (square root of the number of features)
#   By default, max_features is set to auto, which acts the same as sqrt.
# bootstrap: allows us to set whether the subset of observations considered for a tree is created using sampling with
# replacement (the default setting) or without replacement.
# n_estimators: sets the number of decision trees to include in the forest. In Recipe 10.4 we treated n_estimators as
# a hyperparameter and visualized the effect of increasing the number of trees on an evaluation metric.
# n_jobs=-1: using all cores


# Training a Random Forest Regressor

boston = datasets.load_boston()
features = boston.data[:, 0:2]
target = boston.target
# Create random forest classifier object
randomforest = RandomForestRegressor(random_state=0, n_jobs=-1)
# Train model
model = randomforest.fit(features, target)
# Make new observation
observation = [[0.02, 16]]
# Predict observation's value
print(model.predict(observation))


# Identifying Important Features in Random Forests

# Calculate and visualize the importance of each feature
iris = datasets.load_iris()
features = iris.data
target = iris.target
# Create random forest classifier object
randomforest = RandomForestClassifier(random_state=0, n_jobs=-1)
# Train model
model = randomforest.fit(features, target)
# Calculate feature importances
importances = model.feature_importances_
print(importances)
# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]
# Rearrange feature names so they match the sorted feature importances
names = [iris.feature_names[i] for i in indices]
# Create plot
plt.figure()
plt.title("Feature Importance")
# Add bars
plt.bar(range(features.shape[1]), importances[indices])
# Add feature names as x-axis labels
plt.xticks(range(features.shape[1]), names, rotation=90)
# Show plot
# plt.show()


# Selecting Important Features in Random Forests

iris = datasets.load_iris()
features = iris.data
target = iris.target
# Create random forest classifier
randomforest = RandomForestClassifier(random_state=0, n_jobs=-1)
# Create object that selects features with importance greater than or equal to a threshold
selector = SelectFromModel(randomforest, threshold=0.3)
# Feature new feature matrix using selector
features_important = selector.fit_transform(features, target)
# Train random forest using most important featres
model = randomforest.fit(features_important, target)
# Make new observation
observation = [[5,  4]]
# Predict observation's class
print(model.predict(observation))


# Handling Imbalanced Classes

# You have a target vector with highly imbalanced classes and want to train a random forest model
iris = datasets.load_iris()
features = iris.data
target = iris.target
# Make class highly imbalanced by removing first 40 observations
features = features[40:, :]
target = target[40:]
# Create target vector indicating if class 0, otherwise 1
target = np.where((target == 0), 0, 1)
# Create random forest classifier object
randomforest = RandomForestClassifier(random_state=0, n_jobs=-1, class_weight="balanced")
# Train model
model = randomforest.fit(features, target)
# Make new observation
observation = [[5,  4,  3,  2]]
# Predict observation's class
print(model.predict(observation))

# class_weight parameter: if supplied with a dictionary in the form of class names and respective desired weights
# (e.g., {"male": 0.2, "female": 0.8}), RandomForestClassifier will weight the classes accordingly.
# However, often a more useful argument is balanced, wherein classes are automatically weighted inversely proportional
# to how frequently they appear in the data.


# Controlling Tree Size

# You want to manually determine the structure and size of a decision tree
iris = datasets.load_iris()
features = iris.data
target = iris.target
# Create decision tree classifier object
decisiontree = DecisionTreeClassifier(random_state=0,
                                      max_depth=None,
                                      min_samples_split=2,
                                      min_samples_leaf=1,
                                      min_weight_fraction_leaf=0,
                                      max_leaf_nodes=None,
                                      min_impurity_decrease=0)
# Train model
model = decisiontree.fit(features, target)

# max_depth: maximum depth of the tree. If None, the tree is grown until all leaves are pure.
# min_samples_split: minimum number of observations at a node before that node is split.
#   integer: the raw minimum
#   float: the percent of total observations
# min_samples_leaf: min number of observations required to be at a leaf. Uses the same arguments as min_samples_split.
# max_leaf_nodes: maximum number of leaves.
# min_impurity_split: minimum impurity decrease required before a split is performed.

# While it is useful to know these parameters exist, most likely we will only be using max_depth and min_impurity_split
# because shallower trees (sometimes called stumps) are simpler models and thus have lower variance.


# Improving Performance Through Boosting

iris = datasets.load_iris()
features = iris.data
target = iris.target
# Create adaboost tree classifier object
adaboost = AdaBoostClassifier(random_state=0)
# Train model
model = adaboost.fit(features, target)

# In one form of boosting called AdaBoost, we iteratively train a series of weak models (most often a shallow decision
# tree, sometimes called a stump), each iteration giving higher priority to observations the previous model predicted
# incorrectly.

# The end result is an aggregated model where individual weak models focus on more difficult (from a prediction
# perspective) observations. In scikit-learn, we can implement AdaBoost using AdaBoostClassifier or AdaBoostRegressor.
# The most important parameters are base_estimator, n_estimators, and learning_rate:
# base_estimator: learning algorithm to use to train the weak models. This will almost always not need to be changed
#   because by far the most common learner to use with AdaBoost is a decision tree—the parameter’s default argument.
# n_estimators: number of models to iteratively train.
# learning_rate: the contribution of each model to the weights and defaults to 1. Reducing the learning rate will mean
#   the weights will be increased or decreased to a small degree, forcing the model to train slower
#   (but sometimes resulting in better performance scores).
# loss: exclusive to AdaBoostRegressor and sets the loss function to use when updating weights. This defaults to a
# linear loss function, but can be changed to square or exponential.


# Evaluating Random Forests with Out-of-Bag Errors

iris = datasets.load_iris()
features = iris.data
target = iris.target
# Create random tree classifier object
randomforest = RandomForestClassifier(random_state=0, n_estimators=1000, oob_score=True, n_jobs=-1)
# Train model
model = randomforest.fit(features, target)
# View out-of-bag-error
print(randomforest.oob_score_)

# In random forests, each decision tree is trained using a bootstrapped subset of observations. This means that for
# every tree there is a separate subset of observations not being used to train that tree. These are called out-of-bag
# (OOB) observations. We can use OOB observations as a test set to evaluate the performance of our random forest.
