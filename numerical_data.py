import numpy as np
import pandas
from sklearn import preprocessing
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


feature = np.array([[-500.5],
                    [-100.1],
                    [0],
                    [100.1],
                    [900.9]])

# MIN MAX Scaler
minmax_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaled_feature = minmax_scaler.fit_transform(feature)
print(scaled_feature)


# Standardizing scaler
stand_scaler = preprocessing.StandardScaler()
standardized = stand_scaler.fit_transform(feature)  # to have a mean of 0 and a standard deviation of 1
print(standardized)
print("Mean:", round(standardized.mean()))
print("Standard deviation:", standardized.std())

# RobustScaler


# Normalizing observations
features = np.array([[0.5, 0.5],
                     [1.1, 3.4],
                     [1.5, 20.2],
                     [1.63, 34.4],
                     [10.9, 3.3]])

# creating the normalizer
normalizer = preprocessing.Normalizer(norm="l2")
# transforming the feature matrix
print(normalizer.transform(features))

normalizer = preprocessing.Normalizer(norm="l1")  # Manhattan norm; here the simple sum equals to 1, not the square
print(normalizer.transform(features))


# Polynomial and Interaction Features

feature2 = np.array([[2, 3],
                     [2, 3],
                     [2, 3]])

# Create PolynomialFeatures object
polynomial_interaction = preprocessing.PolynomialFeatures(degree=2, include_bias=False)
# degree 2: new features raised to the second power
# degree 3: new features raised to the second and third power
print(polynomial_interaction.fit_transform(feature2))

# interaction: x1*x2; using interaction_only parameter we can have only the interaction
polynomial_interaction = preprocessing.PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
print(polynomial_interaction.fit_transform(feature2))


# Transforming features

# function to add 10 to each element
def add_ten(x):
    return x + 10


ten_transformer = preprocessing.FunctionTransformer(add_ten)
print(ten_transformer.transform(feature2))

# other solution: using pandas' apply
df = pandas.DataFrame(feature2, columns=["feature_1", "feature_2"])
print(df.apply(add_ten))


# Detecting Outliers, extreme observation values

# Create simulated data
features, _ = make_blobs(n_samples=10,
                         n_features=2,
                         centers=1,  # groups
                         random_state=1)
# Replace the first observation's values with extreme values
features[0, 0] = 10000
features[0, 1] = 10000
# Create detector
outlier_detector = EllipticEnvelope(contamination=.1)
# Fit detector
outlier_detector.fit(features)
# Predict outliers
outlier_detector.predict(features)
print(outlier_detector.predict(features))

# A major limitation of this approach is the need to specify a contamination parameter,
# which is the proportion of observations that are outliers—a value that we don’t know.

# Instead of looking at observations as a whole, we can instead look at individual features and
# identify extreme values in those features using interquartile range (IQR):

feature = features[:, 0]
# Create a function to return index of outliers
def indicies_of_outliers(x):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    return np.where((x > upper_bound) | (x < lower_bound))


print(indicies_of_outliers(feature))

# handling outliers
houses = pandas.DataFrame()
houses['Price'] = [534433, 392333, 293222, 4322032]
houses['Bathrooms'] = [2, 3.5, 2, 116]
houses['Square_Feet'] = [1500, 2500, 1500, 48000]
# Filter observations
houses[houses['Bathrooms'] < 20]
# Creating a new feature for outlier:
houses["Outlier"] = np.where(houses["Bathrooms"] < 20, 0, 1)
# adding new feature based on square meter
houses["Log_Of_Square_Feet"] = [np.log(x) for x in houses["Square_Feet"]]
print(houses)


# Discretizating Features (creating categories)

age = np.array([[6],
                [12],
                [20],
                [36],
                [65]])

binarizer = preprocessing.Binarizer(20)
print(binarizer.fit_transform(age))

# using multibple treshholds
print(np.digitize(age, bins=[20, 30, 64], right=True))  # 0 --> LESS than 20; right=True --> LESS THAN OR EQUAL TO 20


# Grouping Observations Using Clustering

# Make simulated feature matrix
features, _ = make_blobs(n_samples=50,
                         n_features=2,
                         centers=3,
                         random_state=1)
dataframe = pandas.DataFrame(features, columns=["feature_1", "feature_2"])
# Make k-means clusterer
clusterer = KMeans(3, random_state=0)
# Fit clusterer
clusterer.fit(features)
# Predict values
dataframe["group"] = clusterer.predict(features)
# View first few observations
print(dataframe.head(10))


# Deleting Observations with Missing Values

features = np.array([[1.1, 11.1],
                     [2.2, 22.2],
                     [3.3, 33.3],
                     [4.4, 44.4],
                     [np.nan, 55]])
# Keep only observations that are not (denoted by ~) missing
print(features[~np.isnan(features).any(axis=1)])

# other solution using pandas
dataframe = pandas.DataFrame(features, columns=["feature_1", "feature_2"])
# Remove observations with missing values
dataframe.dropna()


# Imputing Missing Values

# Make a simulated feature matrix
features, _ = make_blobs(n_samples=1000,
                         n_features=2,
                         random_state=1)
# Standardize the features
scaler = preprocessing.StandardScaler()
standardized_features = scaler.fit_transform(features)

# Replace the first feature's first value with a missing value
true_value = standardized_features[0, 0]
standardized_features[0, 0] = np.nan

# # Create imputer
# # mean_imputer = preprocessing.Imputer(strategy="mean", axis=0)
# # # Impute values
# # features_mean_imputed = mean_imputer.fit_transform(features)
# # # Compare true and imputed values
# # print("True Value:", true_value)
# # print("Imputed Value:", features_mean_imputed[0, 0])
