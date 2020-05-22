import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn import datasets
from sklearn.preprocessing import StandardScaler


# Training a Binary Classifier

iris = datasets.load_iris()
features = iris.data[:100, :]
target = iris.target[:100]
# Standardize features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)
# Create logistic regression object
logistic_regression = LogisticRegression(random_state=0)
# Train model
model = logistic_regression.fit(features_standardized, target)
# Create new observation
new_observation = [[.5, .5, .5, .5]]
# Predict class
print(model.predict(new_observation))
# View predicted probabilities
print(model.predict_proba(new_observation))  # it has 18.8% chance of being class 0 and 81.1% chance of being class 1

# Despite having “regression” in its name, a logistic regression is actually a widely used binary classifier
# (i.e., the target vector can only take two values).


# Training a Multiclass Classifier

# Given more than two classes, you need to train a classifier model
iris = datasets.load_iris()
features = iris.data
target = iris.target
# Standardize features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)
# Create one-vs-rest logistic regression object
logistic_regression = LogisticRegression(random_state=0, multi_class="ovr")  # OVR or MLR
# Train model
model = logistic_regression.fit(features_standardized, target)

# On their own, logistic regressions are only binary classifiers, meaning they cannot handle target vectors with more
# than two classes. However, two clever extensions to logistic regression do just that. First, in one-vs-rest logistic
# regression (OVR) a separate model is trained for each class predicted whether an observation is that class or not
# (thus making it a binary classification problem). It assumes that each classification problem (e.g., class 0 or not)
# is independent.


# Reducing Variance Through Regularization

iris = datasets.load_iris()
features = iris.data
target = iris.target
# Standardize features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)
# Create decision tree classifier object
logistic_regression = LogisticRegressionCV(penalty='l2', Cs=10, random_state=0, n_jobs=-1)
# Train model
model = logistic_regression.fit(features_standardized, target)

# Regularization is a method of penalizing complex models to reduce their variance. Specifically, a penalty term is
# added to the loss function we are trying to minimize, typically the L1 and L2 penalties.

# Higher values of α increase the penalty for larger parameter values (i.e., more complex models). scikit-learn follows
# the common method of using C instead of α where C is the inverse of the regularization strength: C=1α. To reduce
# variance while using logistic regression, we can treat C as a hyperparameter to be tuned to find the value of C that
# creates the best model. In scikit-learn we can use the LogisticRegressionCV class to efficiently tune C.
# LogisticRegressionCV’s parameter, Cs, can either accept a range of values for C to search over (if a list of floats
# is supplied as an argument) or if supplied an integer, will generate a list of that many candidate values drawn from
# a logarithmic scale between –10,000 and 10,000.


# Training a Classifier on Very Large Data

iris = datasets.load_iris()
features = iris.data
target = iris.target
# Standardize features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)
# Create logistic regression object
logistic_regression = LogisticRegression(random_state=0, solver="sag")
# Train model
model = logistic_regression.fit(features_standardized, target)
# scikit-learn’s LogisticRegression offers a number of techniques for training a logistic regression, called solvers.
# Most of the time scikit-learn will select the best solver automatically for us or warn us that we cannot do something
# with that solver. However, there is one particular case we should be aware of.
# stochastic average gradient descent allows us to train a model much faster than other solvers when our data is very
# large. However, it is also very sensitive to feature scaling, so standardizing our features is particularly important.
# We can set our learning algorithm to use this solver by setting solver='sag'.


# Handling Imbalanced Classes

# You need to train a simple classifier model
iris = datasets.load_iris()
features = iris.data
target = iris.target
# Make class highly imbalanced by removing first 40 observations
features = features[40:, :]
target = target[40:]
# Create target vector indicating if class 0, otherwise 1
target = np.where((target == 0), 0, 1)
# Standardize features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)
# Create decision tree classifier object
logistic_regression = LogisticRegression(random_state=0, class_weight="balanced")
# Train model
model = logistic_regression.fit(features_standardized, target)

# Like many other learning algorithms in scikit-learn, LogisticRegression comes with a built-in method of handling
# imbalanced classes. If we have highly imbalanced classes and have not addressed it during preprocessing, we have the
# option of using the class_weight parameter to weight the classes to make certain we have a balanced mix of each class.
# Specifically, the balanced argument will automatically weigh classes inversely proportional to their frequency.
