import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# Fitting a Line

# to train a model that represents a linear relationship between the feature and target vector
boston = load_boston()
features = boston.data[:, 0:2]
target = boston.target
# Create linear regression
regression = LinearRegression()
# Fit the linear regression
model = regression.fit(features, target)
print(model.intercept_)
print(model.coef_)
# In our dataset, the target value is the median value of a Boston home (in the 1970s) in thousands of dollars.
# The price of the first home:
print(target[0]*1000)  # 24000.0
# Predict the target value of the first observation, multiplied by 1000
print(model.predict(features)[0]*1000)  # 24560.23

# Our model’s coefficient of this feature was ~–0.35, meaning that if we multiply this coefficient by 1,000
# (since the target vector is the house price in thousands of dollars), we have the change in house price for each
# additional one crime per capita:
print(model.coef_[0]*1000)  # -349.77
# This says that every single crime per capita will decrease the price of the house by approximately $350!


# Handling Interactive Effects

# You have a feature whose effect on the target variable depends on another feature.
# Create an interaction term to capture that dependence using scikit-learn’s PolynomialFeatures:
boston = load_boston()
features = boston.data[:, 0:2]
target = boston.target
# Create interaction term
interaction = PolynomialFeatures(degree=3, include_bias=False, interaction_only=True)  # degree=3 generates x2 and x3
features_interaction = interaction.fit_transform(features)
# Create linear regression
regression = LinearRegression()
# Fit the linear regression
model = regression.fit(features_interaction, target)

# Sometimes a feature’s effect on our target variable is at least partially dependent on another feature. Imagine a s
# simple coffee-based example where we have two binary features—the presence of sugar and whether or not we have stirred
# and we want to predict if the coffee tastes sweet. Just putting sugar in the coffee (sugar=1, stirred=0) won’t make
# the coffee taste sweet and just stirring the coffee without sugar won’t make it sweet either.

# To create an interaction term, we simply multiply those two values together for every observation:
interaction_term = np.multiply(features[:, 0], features[:, 1])
# View interaction term for first observation
print(interaction_term[0])

# However, while often we will have a substantive reason for believing there is an interaction between two features,
# sometimes we will not. In those cases it can be useful to use PolynomialFeatures to create interaction terms for all
# combinations of features. We can then use model selection strategies to identify the combination of features and
# interaction terms that produce the best model.
# interaction_only=True : to only return interaction terms
# include_bias=False    : by default, PolynomialFeatures would add a feature containing ones called a bias
# degree                : the maximum number of features to create interaction terms from

# We can see the output of PolynomialFeatures matches our manually calculated version:
print(features_interaction[0])


# Fitting a Nonlinear Relationship

boston = load_boston()
features = boston.data[:, 0:1]
target = boston.target
# Create polynomial features x^2 and x^3
polynomial = PolynomialFeatures(degree=3, include_bias=False)  # degree=3 will generate x2 and x3
features_polynomial = polynomial.fit_transform(features)
# Create linear regression
regression = LinearRegression()
# Fit the linear regression
model = regression.fit(features_polynomial, target)

# An example of a linear relationship would be the number of stories a building has and the building’s height. In
# linear regression, we assume the effect of number of stories and building height is approximately constant, meaning a
# 20-story building will be roughly twice as high as a 10-story building.

# We can imagine there is a big difference in test scores between students who study for one hour compared to students
# who did not study at all. However, there is a much smaller difference in test scores between a student who studied
# for 99 hours and a student who studied for 100 hours.
# The effect one hour of studying has on a student’s test score decreases as the number of hours increases.

# To model nonlinear relationships, we can create new features that raise an existing feature up to some power: 2,3,etc.
# The more of these new features we add, the more flexible the “line” created by our model. To make this more explicit,
# imagine we want to create a polynomial to the third degree.
print(features[0])
print(features[0]**2)
print(features[0]**3)
# By including all three features (x, x2, and x3) in our feature matrix and then running a linear regression,
# we have conducted a polynomial regression:
print(features_polynomial[0])


# Reducing Variance with Regularization

# Use a learning algorithm that includes a shrinkage penalty (or regularization) like ridge regression and lasso regr.
boston = load_boston()
features = boston.data
target = boston.target
# Standardize features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)
# Create ridge regression with an alpha value
regression = Ridge(alpha=0.5)
# Fit the linear regression
model = regression.fit(features_standardized, target)

# There are two common types of regularized learners for linear regression: ridge regression and the lasso.
# The only formal difference is the type of shrinkage penalty used. In ridge regression, the shrinkage penalty is a
# tuning hyperparameter multiplied by the squared sum of all coefficients. The lasso is similar, except the shrinkage
# penalty is a tuning hyperparameter multiplied by the sum of the absolute value of all coefficients.

# As a very general rule of thumb, ridge regression often produces slightly better predictions than lasso, but lasso
# produces more interpretable models. If we want a balance between ridge and lasso’s penalty functions we can use
# elastic net, which is simply a regression model with both penalties included.

# The hyperparameter, α, lets us control how much we penalize the coefficients, with higher values of α creating
# simpler models. The ideal value of α should be tuned like any other hyperparameter.
# scikit-learn includes a RidgeCV method that allows us to select the ideal value for α (alpha):
regr_cv = RidgeCV(alphas=[0.1, 1.0, 10.0])
# Fit the linear regression
model_cv = regr_cv.fit(features_standardized, target)
print(model_cv.coef_)
print(model_cv.alpha_)

# Also, because in linear regression the value of the coefficients is partially determined by the scale of the feature,
# and in regularized models all coefficients are summed together, we must standardize the feature prior to training.


# Reducing Features with Lasso Regression

# You want to simplify your linear regression model by reducing the number of features.
boston = load_boston()
features = boston.data
target = boston.target
# Standardize features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)
# Create lasso regression with alpha value
regression = Lasso(alpha=0.5)
# Fit the linear regression
model = regression.fit(features_standardized, target)

# One interesting characteristic of lasso regression’s penalty is that it can shrink the coefficients of a model to
# zero, effectively reducing the number of features in the model. For example, in our solution we set alpha to 0.5 and
# we can see that many of the coefficients are 0, meaning their corresponding features are not used in the model:
print(model.coef_)

# However, if we increase α to a much higher value, we see that literally none of the features are being used:
regression_a10 = Lasso(alpha=10)
model_a10 = regression_a10.fit(features_standardized, target)
print(model_a10.coef_)

# The practical benefit of this effect is that it means that we could include 100 features in our feature matrix and
# then, through adjusting lasso’s α hyperparameter, produce a model that uses only 10 (for instance) of the most
# important features. This lets us reduce variance while improving the interpretability of our model
# (since fewer features is easier to explain).
