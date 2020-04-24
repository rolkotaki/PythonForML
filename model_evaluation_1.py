from sklearn import datasets
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt


# Cross-Validating Models

digits = datasets.load_digits()
features = digits.data
target = digits.target
# Create standardizer
standardizer = StandardScaler()
# Create logistic regression object
logit = LogisticRegression()
# Create a pipeline that standardizes, then runs logistic regression
pipeline = make_pipeline(standardizer, logit)
# Create k-Fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=1)
# Conduct k-fold cross-validation
cv_results = cross_val_score(pipeline,   # Pipeline
                             features,   # Feature matrix
                             target,     # Target vector
                             cv=kf,      # Cross-validation technique
                             scoring="accuracy",  # Loss function
                             n_jobs=-1)  # Use all CPU scores
# Calculate mean
print(cv_results.mean())
print(cv_results)

# KFCV splits the data into k parts called folds. The model is then trained using k – 1 folds—combined into one training
# set—and then the last fold is used as a test set. We repeat this k times, each time using a different fold as the test
# set. The performance on the model for each of the k iterations is then averaged to produce an overall measurement.

# StratifiedKFold: if our target vector contained gender and 80% of the observations were male,
# then each fold would contain 80% male and 20% female observations.


# Creating a Baseline Regression Model

boston = datasets.load_boston()
features, target = boston.data, boston.target
# Make test and training split
features_train, features_test, target_train, target_test = train_test_split(features, target, random_state=0)
# Create a dummy regressor
dummy = DummyRegressor(strategy='mean')
# "Train" dummy regressor
dummy.fit(features_train, target_train)
# Get R-squared score
print(dummy.score(features_test, target_test))

# To compare, we train our model and evaluate the performance score:

# Train simple linear regression model
ols = LinearRegression()
ols.fit(features_train, target_train)
# Get R-squared score
print(ols.score(features_test, target_test))

# DummyRegressor allows us to create a very simple model that we can use as a baseline to compare against our model.


# Creating a Baseline Classification Model

iris = datasets.load_iris()
features, target = iris.data, iris.target
# Split into training and test set
features_train, features_test, target_train, target_test = train_test_split(features, target, random_state=0)
# Create dummy classifier
dummy = DummyClassifier(strategy='uniform', random_state=1)
# "Train" model
dummy.fit(features_train, target_train)
# Get accuracy score
print(dummy.score(features_test, target_test))

# By comparing the baseline classifier to our trained classifier, we can see the improvement:

classifier = RandomForestClassifier()
# Train model
classifier.fit(features_train, target_train)
# Get accuracy score
print(classifier.score(features_test, target_test))

# A common measure of a classifier’s performance is how much better it is than random guessing. DummyClassifier makes
# this comparison easy.
# strategy parameter:
# 'stratified' makes predictions that are proportional to the training set’s target vector’s class proportions
# 'uniform' will generate predictions uniformly at random between the different classes. For example, if 20% of
# observations are women and 80% are men, uniform will produce predictions that are 50% women and 50% men.


# Evaluating Binary Classifier Predictions

# Generate features matrix and target vector
X, y = datasets.make_classification(n_samples=10000,
                                    n_features=3,
                                    n_informative=3,
                                    n_redundant=0,
                                    n_classes=2,
                                    random_state=1)
# Create logistic regression
logit = LogisticRegression()
# Cross-validate model using accuracy
print(cross_val_score(logit, X, y, scoring="accuracy"))   # Accuracy = (TP + TN) / (TP + TN + FP + FN)
# Cross-validate model using
print(cross_val_score(logit, X, y, scoring="precision"))  # Precision = TP / (TP + FP)
# Cross-validate model using recall
print(cross_val_score(logit, X, y, scoring="recall"))     # Recall = TP / (TP + FN)
# ## Recall is the proportion of every positive observation that is truly positive. Recall measures the model’s ability
# ## to identify an observation of the positive class. Models with high recall are optimistic that they have a low bar
# ## for predicting that an observation is in the positive class:
# Cross-validate model using f1
cross_val_score(logit, X, y, scoring="f1")  # F1 = 2 × ( (Precision * Recall) / (Precision + Recall) )
# ## F1 is a balance between precision and recall

# T - true; F - false; P - positive guess; N - negative guess

# Alternatively to using cross_val_score, if we already have the true y values and the predicted y values,
# we can calculate metrics like accuracy and recall directly:

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.1,
                                                    random_state=1)
# Predict values for training target vector
y_hat = logit.fit(X_train, y_train).predict(X_test)
# Calculate accuracy
print(accuracy_score(y_test, y_hat))


# Evaluating Binary Classifier Thresholds

# Receiving Operating Characteristic (ROC):
# ROC compares the presence of true positives and false positives at every probability threshold (i.e., the probability
# at which an observation is predicted to be a class). By plotting the ROC curve, we can see how the model performs.

features, target = datasets.make_classification(n_samples=10000,
                                                n_features=10,
                                                n_classes=2,
                                                n_informative=3,
                                                random_state=3)
# Split into training and test sets
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.1, random_state=1)
# Create classifier
logit = LogisticRegression()
# Train model
logit.fit(features_train, target_train)
# Get predicted probabilities
target_probabilities = logit.predict_proba(features_test)[:, 1]  # ?
# Create true and false positive rates
false_positive_rate, true_positive_rate, threshold = roc_curve(target_test,
                                                               target_probabilities)
# print(logit.classes_)
# print(target_probabilities)[:, 1]

# Plot ROC curve
plt.title("Receiver Operating Characteristic")
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1], c=".7")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

# In this example, the first observation has an ~87% chance of being in the negative class (0) and a 13% chance of
# being in the positive class (1). By default, scikit-learn predicts an observation is part of the positive class if
# the probability is greater than 0.5 (called the threshold). However, instead of a middle ground, we will often want
# to explicitly bias our model to use a different threshold for substantive reasons. For example, if a false positive
# is very costly to our company, we might prefer a model that has a high probability threshold. We fail to predict some
# positives, but when an observation is predicted to be positive, we can be confident that the prediction is correct.

print("Threshold:", threshold[116])  # ???
print("True Positive Rate:", true_positive_rate[116])
print("False Positive Rate:", false_positive_rate[116])

# However, if we increase the threshold to ~80% (i.e., increase how certain the model has to be before it predicts an
# observation as positive) the TPR drops significantly but so does the FPR:

print("Threshold:", threshold[45])
print("True Positive Rate:", true_positive_rate[45])
print("False Positive Rate:", false_positive_rate[45])

# AUCROC: area under the ROC curve; to judge the overall equality of a model at all possible thresholds.
# Calculate area under curve
print(roc_auc_score(target_test, target_probabilities))
