# Using scikit-learn makes it easy to practise ML
# We don't have to worry about loading, transforming or cleaning read datasets

# Some datasets:
# * load_boston: 503 observations on Boston housing prices; good dataset for exploring regression algorithms
# * load_iris:   150 observations on measurements of Iris flowers; good dataset for exploring classification algorithms
# * load_digits: 1.797 observations from images of handwritten digits;  good dataset for teaching image classification

from sklearn import datasets

# Loading datasets

boston = datasets.load_boston()  # loading the dataset
boston_features = boston.data  # creating the feature matrix
boston_target = boston.target  # creating the target vector

# print(boston_features[0])  # first observation
# print(boston_target[0])  # first target


# Creating simulated data


# Regression:
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html#sklearn.datasets.make_regression

# Generate features matrix, target vector, and the true coefficients
features, target, coefficients = datasets.make_regression(n_samples=100,
                                                          n_features=2,
                                                          n_informative=2,
                                                          n_targets=1,
                                                          noise=0.0,
                                                          coef=True,
                                                          random_state=1)

# feature matrix and target vector
print('Feature Matrix\n', features[:3])
print('Target Vector\n', target[:3])
# print(coefficients)

# n_informative: number of features that are used to generate the target vector.
# If n_informative is less than the total number of features (n_features), the resulting dataset will have redundant
# features that can be identified through feature selection techniques.


# Classification
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html#sklearn.datasets.make_classification

# Generate features matrix and target vector
features, target = datasets.make_classification(n_samples=100,
                                                n_features=3,
                                                n_informative=3,
                                                n_redundant=0,
                                                n_classes=2,
                                                weights=[.25, .75],
                                                random_state=1)

# feature matrix and target vector
print('Feature Matrix\n', features[:3])
print('Target Vector\n', target[:3])

# weights: to simulate datasets with imbalanced classes.
# weights = [.25, .75] returns a dataset with 25% of observations from one class and 75% belonging to the second class.


# Clustering
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html#sklearn.datasets.make_blobs

# Generate feature matrix and target vector
features, target = datasets.make_blobs(n_samples=100,
                                       n_features=2,
                                       centers=3,
                                       cluster_std=0.5,
                                       shuffle=True,
                                       random_state=1)

# feature matrix and target vector
print('Feature Matrix\n', features[:3])
print('Target Vector\n', target[:3])

# centers: determines the number of clusters generated
