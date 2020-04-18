from sklearn import datasets, linear_model
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, f_classif, SelectPercentile, RFECV
import pandas as pd
import numpy as np
import warnings


# ##### Dimensionality Reduction Using Feature Selection

# feature selection: selecting high-quality, informative features and dropping less useful features.


# Thresholding Numerical Feature Variance

# removing features with low variance by selecting a subset of features with variances above a given threshold

iris = datasets.load_iris()
features = iris.data
target = iris.target
# Create thresholder
thresholder = VarianceThreshold(threshold=.5)
# Create high variance feature matrix
features_high_variance = thresholder.fit_transform(features)
# View high variance feature matrix
print(features_high_variance[0:3])
# We can see the variance for each feature using variances_:
print(thresholder.fit(features).variances_)

# VT first calculates the variance of each feature, then it drops those whose variance does not meet that threshold.
# If the features have been standardized (mean zero and unit variance), of course variance thresholding will not work.


# Thresholding Binary Feature Variance

# You have a set of binary categorical features and want to remove those with low variance.
# We select a subset of features with a Bernoulli random variable variance above a given threshold.

# Create feature matrix with:
# Feature 0: 80% class 0
# Feature 1: 80% class 1
# Feature 2: 60% class 0, 40% class 1
features = [[0, 1, 0],
            [0, 1, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0]]
# Run threshold by variance
thresholder = VarianceThreshold(threshold=(.75 * (1 - .75)))
print(thresholder.fit_transform(features))


# Handling Highly Correlated Features

# You have a feature matrix and some features are highly correlated. We can use a correlation matrix to check for
# highly correlated features. If highly correlated features exist, consider dropping one of the correlated features.

# Create feature matrix with two highly correlated features
features = np.array([[1, 1, 1],
                     [2, 2, 0],
                     [3, 3, 1],
                     [4, 4, 0],
                     [5, 5, 1],
                     [6, 6, 0],
                     [7, 7, 1],
                     [8, 7, 0],
                     [9, 7, 1]])
# Convert feature matrix into DataFrame
dataframe = pd.DataFrame(features)
# Create correlation matrix
corr_matrix = dataframe.corr().abs()
# Select upper triangle of correlation matrix
# identifying pairs and removing one item of each pair
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
# Drop features
# print(dataframe)
# print(corr_matrix)
# print(upper)
# print(to_drop)
print(dataframe.drop(dataframe.columns[to_drop], axis=1).head(3))


# Removing Irrelevant Features for Classification

# We have a categorical target vector and want to remove uninformative features. If the features are categorical,
# we can calculate a chi-square (χ2) statistic between each feature and the target vector.

iris = datasets.load_iris()
features = iris.data
target = iris.target
# Convert to categorical data by converting data to integers
features = features.astype(int)
# Select two features with highest chi-squared statistics
chi2_selector = SelectKBest(chi2, k=2)
features_kbest = chi2_selector.fit_transform(features, target)
# Show results
# print(features)
# print(features_kbest)
print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_kbest.shape[1])
# we removed two features

# A chi-squared statistic is a single number that tells you how much difference exists between your observed counts
# and the counts you would expect if there were no relationship at all in the population. By calculating the chi-squared
# statistic between a feature and the target vector, we obtain a measurement of the independence between the two.
# chi-squared for feature selection requires that both the target vector and the features are categorical.
# To use our chi-squared approach, all values need to be non-negative.

# OR

# If the features are quantitative, compute the ANOVA F-value between each feature and the target vector
# Select two features with highest F-values
fvalue_selector = SelectKBest(f_classif, k=2)
features_kbest = fvalue_selector.fit_transform(features, target)
# Show results
# print(features_kbest)
print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_kbest.shape[1])

# OR

# Instead of selecting a specific number of features, we can also select the top n percent of features:
# Select top 75% of features with highest F-values
fvalue_selector = SelectPercentile(f_classif, percentile=75)
features_kbest = fvalue_selector.fit_transform(features, target)
# Show results
print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_kbest.shape[1])

# F-value scores examine if, when we group the numerical feature by the target vector, the means for each group are
# significantly different. For example, if we had a binary target vector, gender, and a quantitative feature,
# test scores, the F-value score would tell us if the mean test score for men is different than the mean test score
# for women. If it is not, then test score doesn’t help us predict gender and therefore the feature is irrelevant.


# Recursively Eliminating Features

# We want to automatically select the best features to keep.
# scikit-learn’s RFECV conducts recursive feature elimination (RFE) using cross-validation (CV).
# Repeatedly trains a model, each time removing a feature until model performance (e.g., accuracy) becomes worse.
# The remaining features are the best.

# Suppress an annoying but harmless warning
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
# Generate features matrix, target vector, and the true coefficients
features, target = datasets.make_regression(n_samples=10000, n_features=100, n_informative=2, random_state=1)
# Create a linear regression
ols = linear_model.LinearRegression()
# Recursively eliminate features
rfecv = RFECV(estimator=ols, step=1, scoring="neg_mean_squared_error")
rfecv.fit(features, target)
rfecv.transform(features)

# Number of best features
print(rfecv.n_features_)
# Which categories are best
print(rfecv.support_)
# Rank features best (1) to worst
print(rfecv.ranking_)

# estimator: determines the type of model we want to train (e.g., linear regression)
# step: the number or proportion of features to drop during each loop
# scoring: sets the metric of quality we use to evaluate our model during cross-validation
