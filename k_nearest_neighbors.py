from sklearn import datasets
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV


# The K-Nearest Neighbors classifier (KNN) is one of the simplest yet most commonly used classifiers in supervised
# machine learning. KNN is often considered a lazy learner; it doesn’t technically train a model to make predictions.
# Instead an observation is predicted to be the class of that of the largest proportion of the k nearest observations.
# I.g: if an observation with an unknown class is surrounded by an observation of class 1, it's classified as class 1.


# Finding an Observation’s Nearest Neighbors

iris = datasets.load_iris()
features = iris.data
# Create standardizer
standardizer = StandardScaler()
# Standardize features
features_standardized = standardizer.fit_transform(features)
# Two nearest neighbors
nearest_neighbors = NearestNeighbors(n_neighbors=2).fit(features_standardized)
# Create an observation
new_observation = [1,  1,  1,  1]
# Find distances and indices of the observation's nearest neighbors
distances, indices = nearest_neighbors.kneighbors([new_observation])
# View the nearest neighbors
print(features_standardized[indices])
# The distance variable we created contains the actual distance measurement to each of the two nearest neighbors:
print(distances)

# In our solution we used the dataset of Iris flowers. We created an observation, new_observation, with some values and
# then found the two observations that are closest to our observation. indices contains the locations of the
# observations in our dataset that are closest, so X[indices] displays the values of those observations. Intuitively,
# distance can be thought of as a measure of similarity, so the two closest observations are the two flowers most
# similar to the flower we created.

# Find two nearest neighbors based on euclidean distance
nearestneighbors_euclidean = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(features_standardized)

# More useful info in the book!!!


# Creating a K-Nearest Neighbor Classifier

iris = datasets.load_iris()
X = iris.data
y = iris.target
# Create standardizer
standardizer = StandardScaler()
# Standardize features
X_std = standardizer.fit_transform(X)
# Train a KNN classifier with 5 neighbors
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1).fit(X_std, y)
# Create two observations
new_observations = [[0.75,  0.75,  0.75,  0.75], [1,  1,  1,  1]]
# Predict the class of two observations
print(knn.predict(new_observations))

# KNeighborsClassifier :
# * metric: the distance metric used
# * algorithm: the method used to calculate the nearest neighbors. While there are real differences in the algorithms,
#   by default KNeighborsClassifier attempts to auto-select the best algorithm so often no need to worry about this.
# * weights: if we set this parameter to distance, the closer observations’ votes are weighted more.


# Identifying the Best Neighborhood Size

iris = datasets.load_iris()
features = iris.data
target = iris.target
# Create standardizer
standardizer = StandardScaler()
# Standardize features
features_standardized = standardizer.fit_transform(features)
# Create a KNN classifier
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
# Create a pipeline
pipe = Pipeline([("standardizer", standardizer), ("knn", knn)])
# Create space of candidate values
search_space = [{"knn__n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}]
# Create grid search
classifier = GridSearchCV(pipe, search_space, cv=5, verbose=0).fit(features_standardized, target)

# The size of k has real implications in KNN classifiers. In machine learning we are trying to find a balance between
# bias and variance, and in few places is that as explicit as the value of k. If k = n where n is the number of
# observations, then we have high bias but low variance. If k = 1, we will have low bias but high variance. The best
# model will come from finding the value of k that balances this bias-variance trade-off. In our solution, we used
# GridSearchCV to conduct five-fold cross-validation on KNN classifiers with different values of k. When that is
# completed, we can see the k that produces the best model:
# Best neighborhood size (k)
print(classifier.best_estimator_.get_params()["knn__n_neighbors"])


# Creating a Radius-Based Nearest Neighbor Classifier

# Given an obs. of unknown class, to predict its class based on the class of all observations within a certain distance
iris = datasets.load_iris()
features = iris.data
target = iris.target
# Create standardizer
standardizer = StandardScaler()
# Standardize features
features_standardized = standardizer.fit_transform(features)
# Train a radius neighbors classifier
rnn = RadiusNeighborsClassifier(radius=.5, n_jobs=-1).fit(features_standardized, target)
# Create two observations
new_observations = [[1,  1,  1,  1]]
# Predict the class of two observations
print(rnn.predict(new_observations))

# First, in RadiusNeighborsClassifier we need to specify the radius of the fixed area used to determine if an
# observation is a neighbor using radius. Unless there is some substantive reason for setting radius to some value,
# it is best to treat it like any other hyperparameter and tune it during model selection. The second useful parameter
# is outlier_label, which indicates what label to give an observation that has no observations within the radius—which
# itself can often be a useful tool for identifying outliers.
