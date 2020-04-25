import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from scipy.stats import uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Search statements are commented from after the first part so that they are not executed as it takes a little more time
# Some issues found during learning:
# https://stackoverflow.com/questions/57153111/scipy-optmize-minimize-iteration-limit-exceeded
# https://stackoverflow.com/questions/60868629/valueerror-solver-lbfgs-supports-only-l2-or-none-penalties-got-l1-penalty


# Selecting Best Models Using Exhaustive Search

# You want to select the best model by searching over a range of hyperparameters
iris = datasets.load_iris()
features = iris.data
target = iris.target
# Create logistic regression
logistic = linear_model.LogisticRegression()
# Create range of candidate penalty hyperparameter values
penalty = ['l1', 'l2']
# Create range of candidate regularization hyperparameter values
C = np.logspace(0, 4, 10)
# Create dictionary hyperparameter candidates
hyperparameters = dict(C=C, penalty=penalty)
# Create grid search
gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)  # verbose: nb of messages during the search (0-3)
# Fit grid search
best_model = gridsearch.fit(features, target)

# For C, we define 10 possible values: np.logspace(0, 4, 10). Also, we define two possible values for the regularization
# penalty: ['l1', 'l2']. For each combination of C and regularization penalty values, we train the model and evaluate it
# using k-fold cross-validation. In our solution, we had 10 possible values of C, 2 possible values of regularization
# penalty, and 5 folds. They created 10 × 2 × 5 = 100 candidate models from which the best was selected.
# Once GridSearchCV is complete, we can see the hyperparameters of the best model:
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])
# Predict target vector
print(best_model.predict(features))


# Selecting Best Models Using Randomized Search

# You want a computationally cheaper method than exhaustive search to select the best model
iris = datasets.load_iris()
features = iris.data
target = iris.target
# Create logistic regression
logistic = linear_model.LogisticRegression()
# Create range of candidate regularization penalty hyperparameter values
penalty = ['l1', 'l2']
# Create distribution of candidate regularization hyperparameter values
C = uniform(loc=0, scale=4)
# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)
# Create randomized search
# n_iter: number of sampled combinations of hyperparameters
randomizedsearch = RandomizedSearchCV(logistic, hyperparameters, random_state=1, n_iter=100, cv=5, verbose=0, n_jobs=-1)
# Fit randomized search
# best_model = randomizedsearch.fit(features, target)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# GridSearchCV uses a user-defined set of hyperparameter values to search for the best model. A more efficient method
# than GridSearchCV’s brute-force search is to search over a specific number of random combinations of hyperparameter
# values from user-supplied distributions (e.g., normal, uniform).
# scikit-learn implements this randomized search technique with RandomizedSearchCV.
# With RandomizedSearchCV, if we specify a distribution, scikit-learn will randomly sample without replacement
# hyperparameter values from that. Here we randomly sample 10 values from a uniform distribution ranging from 0 to 4.
# View best hyperparameters
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])
# Predict target vector
print(best_model.predict(features))


# Selecting Best Models from Multiple Learning Algorithms

# Select the best model by searching over a range of learning algorithms and their respective hyperparameters
# Set random seed
np.random.seed(0)
# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target
# Create a pipeline
pipe = Pipeline([("classifier", RandomForestClassifier())])
# Create dictionary with candidate learning algorithms and their hyperparameters
search_space = [{"classifier": [linear_model.LogisticRegression()],
                 "classifier__penalty": ['l1', 'l2'],
                 "classifier__C": np.logspace(0, 4, 10)},
                {"classifier": [RandomForestClassifier()],
                 "classifier__n_estimators": [10, 100, 1000],
                 "classifier__max_features": [1, 2, 3]}]
# Create grid search
gridsearch = GridSearchCV(pipe, search_space, cv=5, verbose=0)
# Fit grid search
# best_model = gridsearch.fit(features, target)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# After the search, we can use best_estimator_ to view the best model’s learning algorithm and hyperparameters:
# print(best_model.best_estimator_.get_params()["classifier"])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Predict target vector
print(best_model.predict(features))


# Selecting Best Models When Preprocessing

# You want to include a preprocessing step during model selection -->
# Create a pipeline that includes the preprocessing step and any of its parameters
# Set random seed
np.random.seed(0)
# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target
# Create a preprocessing object that includes StandardScaler features and PCA
preprocess = FeatureUnion([("std", StandardScaler()), ("pca", PCA())])
# Create a pipeline
pipe = Pipeline([("preprocess", preprocess),
                 ("classifier", linear_model.LogisticRegression())])
# Create space of candidate values
search_space = [{"preprocess__pca__n_components": [1, 2, 3],
                 "classifier__penalty": ["l1", "l2"],
                 "classifier__C": np.logspace(0, 4, 10)}]
# Create grid search
clf = GridSearchCV(pipe, search_space, cv=5, verbose=0, n_jobs=-1)
# Fit grid search
# best_model = clf.fit(features, target)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Very often we will need to preprocess our data before using it to train a model. ... For this reason, we cannot
# preprocess the data and then run GridSearchCV. Rather, the preprocessing steps must be a part of the set of actions
# taken by GridSearchCV. FeatureUnion allows us to combine multiple preprocessing actions properly. In our solution
# we use FeatureUnion to combine two preprocessing steps: standardize the feature values (StandardScaler) and
# Principal Component Analysis (PCA). We then include preprocess into a pipeline with our learning algorithm.
# Second, some preprocessing methods have their own parameters, which often have to be supplied by the user.
# For example, dimensionality reduction using PCA requires the user to define the number of principal components to use
# to produce the transformed feature set. Ideally, we would choose the number of components that produces a model with
# the greatest performance for some evaluation test metric. Luckily, scikit-learn makes this easy. When we include
# candidate component values in the search space, they are treated like any other hyperparameter to be searched over.
# In our solution, we defined features__pca__n_components': [1, 2, 3] in the search space to indicate that we wanted to
# discover if one, two, or three principal components produced the best model.

# After model selection is complete, we can view the preprocessing values that produced the best model. For example,
# we can see the best number of principal components:
# print(best_model.best_estimator_.get_params()['preprocess__pca__n_components'])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# Speeding Up Model Selection with Parallelization

# Use all the cores in your machine by setting n_jobs=-1 :
# GridSearchCV(pipe, search_space, cv=5, verbose=1, n_jobs=-1)

# The parameter n_jobs defines the number of models to train in parallel.


# Speeding Up Model Selection Using Algorithm-Specific Methods

# If you are using a select number of learning algorithms, use scikit-learn’s model-specific
# cross-validation hyperparameter tuning. For example, LogisticRegressionCV:
iris = datasets.load_iris()
features = iris.data
target = iris.target
# Create cross-validated logistic regression
logit = linear_model.LogisticRegressionCV(Cs=100)
# Train model
logit.fit(features, target)
print(logit)

# Sometimes the characteristics of a learning algorithm allow us to search for the best hyperparameters significantly
# faster than either brute force or randomized model search methods. In scikit-learn, many learning algorithms
# (e.g., ridge, lasso, and elastic net regression) have an algorithm-specific cross-validation method to take advantage
# of this. For example, LogisticRegression is used to conduct a standard logistic regression classifier, while
# LogisticRegressionCV implements an efficient cross-validated logistic regression classifier that has the ability
# to identify the optimum value of the hyperparameter C.
#
# LogisticRegressionCV method includes a parameter Cs. If supplied a list, Cs is the candidate hyperparameter values
# to select from. If supplied an integer, the parameter Cs generates a list of that many candidate values. The candidate
# values are drawn logarithmically from a range between 0.0001 and 1,0000 (a range of reasonable values for C).


# Evaluating Performance After Model Selection

# You want to evaluate the performance of a model found through model selection
iris = datasets.load_iris()
features = iris.data
target = iris.target
# Create logistic regression
logistic = linear_model.LogisticRegression()
# Create range of 20 candidate values for C
C = np.logspace(0, 4, 20)
# Create hyperparameter options
hyperparameters = dict(C=C)
# Create grid search
gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, n_jobs=-1, verbose=1)
# Conduct nested cross-validation and outut the average score
print(cross_val_score(gridsearch, features, target).mean())

# In nested cross-validation, the “inner” cross-validation selects the best model, while the “outer” cross-validation
# provides us with an unbiased evaluation of the model’s performance. In our solution, the inner cross-validation is o
# ur GridSearchCV object, which we then wrap in an outer cross-validation using cross_val_score.

best_model = gridsearch.fit(features, target)
# From the output you can see the inner cross-validation trained 20 candidate models five times, totaling 100 models.
# Next, nest clf inside a new cross-validation, which defaults to three folds:
scores = cross_val_score(gridsearch, features, target)
print(scores)
