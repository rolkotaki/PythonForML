from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA, NMF, TruncatedSVD
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.sparse import csr_matrix


# ##### Dimensionality Reduction Using Feature Extraction


# Reducing Features Using Principal Components - reduce the number of features while retaining the variance in the data

# Load the data
digits = datasets.load_digits()
# Standardize the feature matrix
features = StandardScaler().fit_transform(digits.data)
# Create a PCA that will retain 99% of variance
pca = PCA(n_components=0.99, whiten=True)
# Conduct PCA
features_pca = pca.fit_transform(features)
# Show results
print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_pca.shape[1])

# n_components:
# if n_components > 1, n_components will return that many features
# if n_components is between 0 and 1, pca returns the minimum amount of features that retain that much variance
# It is common to use values of 0.95 and 0.99, meaning 95% and 99% of the variance of the original features
# whiten=True transforms the values of each principal component so that they have zero mean and unit variance

# http://www.math.union.edu/~jaureguj/PCA.pdf


# Reducing Features When Data Is Linearly Inseparable

# You suspect you have linearly inseparable data and want to reduce the dimensions
# Using an extension of principal component analysis that uses kernels to allow for non-linear dimensionality reduction

# Create linearly inseparable data
features, _ = datasets.make_circles(n_samples=1000, random_state=1, noise=0.1, factor=0.1)
# Apply kernal PCA with radius basis function (RBF) kernel
kpca = KernelPCA(kernel="rbf", gamma=15, n_components=1)
features_kpca = kpca.fit_transform(features)
print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_kpca.shape[1])

# If the data is linearly separable (i.e., you can draw a straight line or hyperplane between different classes)
# then PCA works well. However, if your data is not linearly separable (e.g., you can only separate classes
# using a curved decision boundary), the linear transformation will not work as well.

# http://sebastianraschka.com/Articles/2014_kernel_pca.html


# Reducing Features by Maximizing Class Separability

iris = datasets.load_iris()
features = iris.data
target = iris.target
# Create and run an LDA (linear discriminant analysis), then use it to transform the features
lda = LinearDiscriminantAnalysis(n_components=1)
features_lda = lda.fit(features, target).transform(features)
# Print the number of features
print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_lda.shape[1])

# We can use explained_variance_ratio_ to view the amount of variance explained by each component.
# In our solution the single component explained over 99% of the variance:
print(lda.explained_variance_ratio_)

# In PCA we were only interested in the component axes that maximize the variance in the data,
# while in LDA we have the additional goal of maximizing the differences between classes.

# n_components, indicating the number of features we want returned
# We can run LinearDiscriminantAnalysis with n_components set to None to return the ratio of variance
# explained by every component feature, then calculate how many components are required to get above some
# threshold of variance explained (often 0.95 or 0.99):

# Create and run LDA
lda = LinearDiscriminantAnalysis(n_components=None)
features_lda = lda.fit(features, target)
# Create array of explained variance ratios
lda_var_ratios = lda.explained_variance_ratio_


def select_n_components(var_ratio, goal_var: float) -> int:
    # Set initial variance explained so far
    total_variance = 0.0
    # Set initial number of features
    n_components = 0
    # For the explained variance of each feature:
    for explained_variance in var_ratio:
        # Add the explained variance to the total
        total_variance += explained_variance
        # Add one to the number of components
        n_components += 1
        # If we reach our goal level of explained variance
        if total_variance >= goal_var:
            # End the loop
            break
    # Return the number of components
    return n_components


# Run function
print(select_n_components(lda_var_ratios, 0.95))
print(lda_var_ratios)

# http://sebastianraschka.com/Articles/2014_python_lda.html


# Reducing Features Using Matrix Factorization

# using non-negative matrix factorization (NMF) to reduce the dimensionality of the feature matrix

digits = datasets.load_digits()

# Load feature matrix
features = digits.data
# Create, fit, and apply NMF
nmf = NMF(n_components=10, random_state=1)
features_nmf = nmf.fit_transform(features)
# Show results
print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_nmf.shape[1])

# NMF is an unsupervised technique for linear dimensionality reduction that factorizes (i.e., breaks up into
# multiple matrices whose product approximates the original matrix) the feature matrix into matrices representing
# the latent relationship between observations and their features.
# Formally, given a desired number of returned features, r, NMF factorizes our feature matrix such that:
#                      V≈WH
# where V is our d × _n feature matrix (i.e., d features, n observations), W is a d × r, and H is an r × n matrix.
# By adjusting the value of r we can set the amount of dimensionality reduction desired.

# One major requirement of NMA is that the feature matrix cannot contain negative values.


# Reducing Features on Sparse Data

# Truncated Singular Value Decomposition (TSVD) when we have a sparse feature matrix and want to reduce dimensionality

digits = datasets.load_digits()
# Standardize feature matrix
features = StandardScaler().fit_transform(digits.data)
# Make sparse matrix
features_sparse = csr_matrix(features)
# Create a TSVD
tsvd = TruncatedSVD(n_components=10)
# Conduct TSVD on sparse matrix
features_sparse_tsvd = tsvd.fit(features_sparse).transform(features_sparse)
# Show results
print("Original number of features:", features_sparse.shape[1])
print("Reduced number of features:", features_sparse_tsvd.shape[1])

# TSVD provides us with the ratio of the original feature matrix’s variance explained by each component,
# we can select the number of components that explain a desired amount of variance (95% or 99% are common values).
# In our solution the first three outputted components explain approximately 30% of the original data’s variance:
# Sum of first three components' explained variance ratios
print(tsvd.explained_variance_ratio_[0:3].sum())

# automating process to find n_components:

# Create and run an TSVD with one less than number of features
tsvd = TruncatedSVD(n_components=features_sparse.shape[1]-1)
features_tsvd = tsvd.fit(features)
# List of explained variances
tsvd_var_ratios = tsvd.explained_variance_ratio_


def select_n_components_tvsd(var_ratio, goal_var):
    # Set initial variance explained so far
    total_variance = 0.0
    # Set initial number of features
    n_components = 0
    # For the explained variance of each feature:
    for explained_variance in var_ratio:
        # Add the explained variance to the total
        total_variance += explained_variance
        # Add one to the number of components
        n_components += 1
        # If we reach our goal level of explained variance
        if total_variance >= goal_var:
            # End the loop
            break
    # Return the number of components
    return n_components


print(select_n_components_tvsd(tsvd_var_ratios, 0.95))

# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#sklearn.decomposition.TruncatedSVD
