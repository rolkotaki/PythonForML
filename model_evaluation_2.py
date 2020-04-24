from sklearn.model_selection import cross_val_score, train_test_split, learning_curve, validation_curve
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, silhouette_score, make_scorer, r2_score, classification_report
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# Evaluating Multiclass Classifier Predictions

# cross-validation with an evaluation metric capable of handling more than two classes
features, target = datasets.make_classification(n_samples=10000,
                                                n_features=3,
                                                n_informative=3,
                                                n_redundant=0,
                                                n_classes=3,
                                                random_state=1)
# Create logistic regression
logit = LogisticRegression()
# Cross-validate model using accuracy
print(cross_val_score(logit, features, target, scoring='accuracy'))
# we have a balanced accuracy

# Cross-validate model using macro averaged F1 score (explanation for this in model_evaluation_1 also)
print(cross_val_score(logit, features, target, scoring='f1_macro'))
# macro:    mean of metric scores for each class, weighting each class equally
# weighted: mean of metric scores for each class, weighting each class proportional to its size in the data
# micro:    mean of metric scores for each observation-class combination


# Visualizing a Classifier’s Performance

# Use a confusion matrix, which compares predicted classes and true classes
iris = datasets.load_iris()
features = iris.data
target = iris.target
# Create list of target class names
class_names = iris.target_names
# Create training and test set
features_train, features_test, target_train, target_test = train_test_split(
    features, target, random_state=1)
# Create logistic regression
classifier = LogisticRegression()
# Train model and make predictions
target_predicted = classifier.fit(features_train, target_train).predict(features_test)
# Create confusion matrix
matrix = confusion_matrix(target_test, target_predicted)
# Create pandas dataframe
dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)
# Create heatmap
sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues")
plt.title("Confusion Matrix"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()

# In the solution, the top-left cell is the number of observations predicted to be Iris setosa (indicated by the column)
# that are actually Iris setosa (indicated by the row). The models accurately predicted all Iris setosa flowers.


# Evaluating Regression Models

# Using mean squared error (MSE)
features, target = datasets.make_regression(n_samples=100,
                                            n_features=3,
                                            n_informative=3,
                                            n_targets=1,
                                            noise=50,
                                            coef=False,
                                            random_state=1)
# Create a linear regression object
ols = LinearRegression()
# Cross-validate the linear regression using (negative) MSE
print(cross_val_score(ols, features, target, scoring='neg_mean_squared_error'))
# Another common regression metric is the coefficient of determination R-squared
# Cross-validate the linear regression using R-squared
print(cross_val_score(ols, features, target, scoring='r2'))

# The higher the value of MSE, the greater the total squared error and thus the worse the model.
# There are a number of mathematical benefits to squaring the error term, including that it forces all error values
# to be positive, but one often unrealized implication is that squaring penalizes a few large errors more than many
# small errors, even if the absolute value of the errors is the same.
# For example, imagine two models, A and B, each with two observations:
# Model A has errors of 0 and 10 and thus its MSE is 02 + 102 = 100.   # 2 means square
# Model B has two errors of 5 each, and thus its MSE is 52 + 52 = 50.
# Both models have the same total error, 10; however, MSE would consider Model A worse than Model B (MSE = 50).
# In practice this implication is rarely an issue and MSE works perfectly fine as an evaluation metric.


# Evaluating Clustering Models

# You have used an unsupervised learning algorithm to cluster your data and want to know how well it did.
# One option is to evaluate clustering using silhouette coefficients, which measure the quality of the clusters.
features, _ = datasets.make_blobs(n_samples=1000,
                                  n_features=10,
                                  centers=2,
                                  cluster_std=0.5,
                                  shuffle=True,
                                  random_state=1)
# Cluster data using k-means to predict classes
model = KMeans(n_clusters=2, random_state=1).fit(features)
# Get predicted classes
target_predicted = model.labels_
# Evaluate model
print(silhouette_score(features, target_predicted))

# While we cannot evaluate predictions versus true values if we don’t have a target vector, we can evaluate the nature
# of the clusters themselves. Intuitively, we can imagine “good” clusters having very small distances between
# observations in the same cluster (i.e., dense clusters) and large distances between the different clusters.
# The value returned by silhouette_score is the mean silhouette coefficient for all observations.
# Silhouette coefficients range between –1 and 1, with 1 indicating dense, well-separated clusters.


# Creating a Custom Evaluation Metric

# Create the metric as a function and convert it into a scorer function using scikit-learn’s make_scorer
features, target = datasets.make_regression(n_samples=100,
                                            n_features=3,
                                            random_state=1)
# Create training set and test set
features_train, features_test, target_train, target_test = train_test_split(
     features, target, test_size=0.10, random_state=1)


# Create custom metric
def custom_metric(target_test, target_predicted):
    # Calculate r-squared score
    r2 = r2_score(target_test, target_predicted)
    # Return r-squared score
    return r2


# Make scorer and define that higher scores are better
score = make_scorer(custom_metric, greater_is_better=True)
# Create ridge regression object
classifier = Ridge()
# Train ridge regression model
model = classifier.fit(features_train, target_train)
# Apply custom scorer
print(score(model, features_test, target_test))

# We define a function that takes in two arguments—the ground truth target vector and our predicted values—and outputs
# some score. Second, we use make_scorer to create a scorer object, making sure to specify whether higher or
# lower scores are desirable


# Visualizing the Effect of Training Set Size

# to evaluate the effect of the number of observations in your training set on some metric
# They are commonly used to determine if our learning algorithms would benefit from gathering additional training data.
digits = datasets.load_digits()
features, target = digits.data, digits.target
# Create CV training and test scores for various training set sizes
train_sizes, train_scores, test_scores = learning_curve(# Classifier
                                                        RandomForestClassifier(),
                                                        # Feature matrix
                                                        features,
                                                        # Target vector
                                                        target,
                                                        # Number of folds
                                                        cv=10,
                                                        # Performance metric
                                                        scoring='accuracy',
                                                        # Use all computer cores
                                                        n_jobs=-1,
                                                        # Sizes of 50
                                                        # training set
                                                       train_sizes=np.linspace(
                                                       0.01,  # from 1% of observations
                                                       1.0,   # to 100% of observations
                                                       50))   # 50 training sets
# Create means and standard deviations of training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
# Create means and standard deviations of test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Draw lines
plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")
# Draw bands
plt.fill_between(train_sizes, train_mean - train_std,
                 train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std,
                 test_mean + test_std, color="#DDDDDD")
# Create plot
plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"),
plt.legend(loc="best")
plt.tight_layout()
plt.show()


# Creating a Text Report of Evaluation Metrics

# You want a quick description of a classifier’s performance
iris = datasets.load_iris()
features = iris.data
target = iris.target
# Create list of target class names
class_names = iris.target_names
# Create training and test set
features_train, features_test, target_train, target_test = train_test_split(features, target, random_state=1)
# Create logistic regression
classifier = LogisticRegression()
# Train model and make predictions
model = classifier.fit(features_train, target_train)
target_predicted = model.predict(features_test)
# Create a classification report
print(classification_report(target_test,
                            target_predicted,
                            target_names=class_names))


# Visualizing the Effect of Hyperparameter Values

# You want to understand how the performance of a model changes as the value of some hyperparameter changes
digits = datasets.load_digits()
features, target = digits.data, digits.target
# Create range of values for parameter
param_range = np.arange(1, 250, 2)
# Calculate accuracy on training and test set using range of parameter values
train_scores, test_scores = validation_curve(
                                # Classifier
                                RandomForestClassifier(),
                                # Feature matrix
                                features,
                                # Target vector
                                target,
                                # Hyperparameter to examine
                                param_name="n_estimators",  # name of the hyperparameter to vary
                                # Range of hyperparameter's values
                                param_range=param_range,    # value of the hyperparameter to use
                                # Number of folds
                                cv=3,
                                # Performance metric
                                scoring="accuracy",         # evaluation metric used to judge to model
                                # Use all computer cores
                                n_jobs=-1)
# Calculate mean and standard deviation for training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
# Calculate mean and standard deviation for test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot mean accuracy scores for training and test sets
plt.plot(param_range, train_mean, label="Training score", color="black")
plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")
# Plot accurancy bands for training and test sets
plt.fill_between(param_range, train_mean - train_std,
                 train_mean + train_std, color="gray")
plt.fill_between(param_range, test_mean - test_std,
                 test_mean + test_std, color="gainsboro")
# Create plot
plt.title("Validation Curve With Random Forest")
plt.xlabel("Number Of Trees")
plt.ylabel("Accuracy Score")
plt.tight_layout()
plt.legend(loc="best")
plt.show()

# Most training algorithms contain hyperparameters that must be chosen before the training process begins.
# For example, a random forest classifier creates a “forest” of decision trees, each of which votes on the predicted
# class of an observation. One hyperparameter in random forest classifiers is the number of trees in the forest.
# Most often hyperparameter values are selected during model selection. However, it is occasionally useful to visualize
# how model performance changes as the hyperparameter value changes.
# In our solution, we plot the changes in accuracy for a random forest classifier for the training set and during
# cross-validation as the number of trees increases. When we have a small number of trees, both the training and
# cross-validation score are low, suggesting the model is underfitted. As the number of trees increases to 250, the
# accuracy of both levels off, suggesting there is not much value in the computational cost of training a massive forest
