import numpy as np
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.neighbors import KNeighborsClassifier
import pandas
# from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris


# nominal category: with no intrinsic ordreing; i.eg.: blue, red, green; man, woman; banana, appel
# ordinal category: having natural ordering; i.eg.: low, medium, high; young, old; agree, neutral, disagree


# Nominal Categorical Features

# one-hot encoding values (also called dummying)
feature = np.array([["Texas"],
                    ["California"],
                    ["Texas"],
                    ["Delaware"],
                    ["Texas"]])
# Create one-hot encoder
one_hot = LabelBinarizer()
# One-hot encode feature
print(one_hot.fit_transform(feature))
print(one_hot.classes_)

# reversing the one-hot encoding
one_hot.inverse_transform(one_hot.transform(feature))

# handling observations with multiple classes
multiclass_feature = [("Texas", "Florida"),
                      ("California", "Alabama"),
                      ("Texas", "Florida"),
                      ("Delware", "Florida"),
                      ("Texas", "Alabama")]
# Create multiclass one-hot encoder
one_hot_multiclass = MultiLabelBinarizer()
# One-hot encode multiclass feature
one_hot_multiclass.fit_transform(multiclass_feature)

# https://stats.stackexchange.com/questions/231285/dropping-one-of-the-columns-when-using-one-hot-encoding


# Encoding Ordinal Categorical Features

dataframe = pandas.DataFrame({"Score": ["Low", "Low", "Medium", "Medium", "High"]})
# Create mapper
scale_mapper = {"Low": 1,
                "Medium": 2,
                "High": 3}
# Replace feature values with scale
print(dataframe["Score"].replace(scale_mapper))


# Encoding Dictionaries of Features

# Create dictionary
data_dict = [{"Red": 2, "Blue": 4},
             {"Red": 4, "Blue": 3},
             {"Red": 1, "Yellow": 2},
             {"Red": 2, "Yellow": 2}]
# Create dictionary vectorizer
dictvectorizer = DictVectorizer(sparse=False)  # True: useful with big matrices to decrease memory space
# Convert dictionary to feature matrix
features = dictvectorizer.fit_transform(data_dict)
print(features)

# checking feature names
feature_names = dictvectorizer.get_feature_names()
print(feature_names)


# Imputing Missing Class Values

# The ideal solution is to train a machine learning classifier algorithm to predict the missing values,
# commonly a k-nearest neighbors (KNN) classifier.
# KNN: assigns to the missing value the median class of the k nearest observations

# Create feature matrix with categorical feature
X = np.array([[0, 2.10, 1.45],
              [1, 1.18, 1.33],
              [0, 1.22, 1.27],
              [1, -0.21, -1.19]])
# Create feature matrix with missing values in the categorical feature
X_with_nan = np.array([[np.nan, 0.87, 1.31],
                       [np.nan, -0.67, -0.22]])

print(X[:, 1:])
print(X[:, 0])

# Train KNN learner
clf = KNeighborsClassifier(3, weights='distance')
trained_model = clf.fit(X[:, 1:], X[:, 0])

# Predict missing values' class
imputed_values = trained_model.predict(X_with_nan[:, 1:])
# Join column of predicted class with their other features
X_with_imputed = np.hstack((imputed_values.reshape(-1, 1), X_with_nan[:, 1:]))
print(X_with_imputed)
# Join two feature matrices
print(np.vstack((X_with_imputed, X)))

# An alternative solution is to fill in missing values with the feature’s most frequent value.

# Join the two feature matrices
# X_complete = np.vstack((X_with_nan, X))
# imputer = Imputer(strategy='most_frequent', axis=0)
# imputer.fit_transform(X_complete)

# Imputer does not work, need to check!!!


# Handling Imbalanced Classes

# To demonstrate our solutions, we need to create some data with imbalanced classes. Fisher’s Iris dataset contains
# three balanced classes of 50 observations, each indicating the species of flower (Iris setosa, Iris virginica,
# and Iris versicolor). To unbalance the dataset, we remove 40 of the 50 Iris setosa observations and
# then merge the Iris virginica and Iris versicolor classes. The end result is a binary target vector indicating
# if an observation is an Iris setosa flower or not. The result is 10 observations of Iris setosa (class 0)
# and 100 observations of not Iris setosa (class 1):

# Load iris data
iris = load_iris()
# Create feature matrix
features = iris.data
# Create target vector
target = iris.target
# Remove first 40 observations
features = features[40:, :]
target = target[40:]

# Create binary target vector indicating if class 0
target = np.where((target == 0), 0, 1)
# imbalanced target vector
print(target)

# RandomForestClassifier is a popular classification algorithm and includes a class_weight parameter.
# You can pass an argument specifying the desired class weights explicitly

weights = {0: .9, 1: 0.1}
# Create random forest classifier with weights
RandomForestClassifier(class_weight=weights)
# OR
# Train a random forest with balanced class weights
RandomForestClassifier(class_weight="balanced")

# In downsampling we create a random subset of the majority class of equal size to the minority class.
# In upsampling we repeatedly sample with replacement from the minority class to make it of equal size as the majority class.

# Indicies of each class' observations
i_class0 = np.where(target == 0)[0]
i_class1 = np.where(target == 1)[0]
# Number of observations in each class
n_class0 = len(i_class0)
n_class1 = len(i_class1)
# For every observation of class 0, randomly sample from class 1 without replacement
i_class1_downsampled = np.random.choice(i_class1, size=n_class0, replace=False)

# Join together class 0's target vector with the downsampled class 1's target vector
print(np.hstack((target[i_class0], target[i_class1_downsampled])))

# Join together class 0's feature matrix with the downsampled class 1's feature matrix
print(np.vstack((features[i_class0, :], features[i_class1_downsampled, :])))

# other option is to upsample the minority class

# For every observation in class 1, randomly sample from class 0 with replacement
i_class0_upsampled = np.random.choice(i_class0, size=n_class1, replace=True)
# Join together class 0's upsampled target vector with class 1's target vector
np.concatenate((target[i_class0_upsampled], target[i_class1]))

# Join together class 0's upsampled feature matrix with class 1's feature matrix
np.vstack((features[i_class0_upsampled, :], features[i_class1, :]))[0:5]
