import numpy as np
from scipy import sparse


# *** Vector

vector_row = np.array([1, 2, 3])
vector_column = np.array([[1], [2], [3]])

print("VECTOR")
print(vector_row[0])     # 1
print(vector_column[2])  # [3]
vector_row[:]     # all elements of the vector; [1 2 3]
vector_row[1:]    # all elements after the first; [2 3]
vector_row[:1]    # all elements up to and including the first; [1]
vector_row[-1]    # last element; 3

np.dot(vector_row, vector_column)  # dot product of the two vectors; 14 (1*1 + 2*2 + 3*3)

print(np.array([vector_row]).T)  # transposing row vector to column vector


# *** Matrix

matrix = np.mat([[1, 2], [3, 4], [5, 6]])  # a two dimensional array

print("\nMATRIX")
print(matrix[0, 0])    # 1
print(matrix[2, 1])    # 6
matrix[:2, :]  # the first two rows and all columns of a matrix
matrix[:, 1:2]  # all rows and the second column

matrix.shape  # shape of the matrix --> (3, 2) // 3 rows and 2 columns
matrix.size   # 6
matrix.ndim   # 2 dimensions

np.max(matrix)  # 6
np.min(matrix)  # 1
np.max(matrix, axis=0)  # maximum element in each column  // [[5 6]]
np.max(matrix, axis=1)  # maximum element in each row

np.mean(matrix)  # average of the matrix; 3.5
np.var(matrix)   # variance of the matrix; 2.916*
np.std(matrix)   # standard deviation; 1.7078
# just like with max() before, we can calculate these for columns or rows only

print(matrix.T)  # transposing matrix; column and row indices of each element are swapped
matrix.flatten()  # transforming the matrix into a one-dimensional array

matrix.diagonal()
matrix.diagonal(offset=1)      # diagonal one above the main
matrix.diagonal(offset=-1)     # diagonal one below the main
matrix.trace()                 # trace of matrix = sum of diagonal elements
np.linalg.matrix_rank(matrix)  # rank of the matrix; https://en.wikipedia.org/wiki/Rank_(linear_algebra)


matrix = np.array([[1, -1, 3],
                   [1, 1, 6],
                   [3, 8, 9]])

np.linalg.det(matrix)          # determinant of the SQUARE matrix; https://en.wikipedia.org/wiki/Determinant

# eigenvalues and eigenvectors of the SQUARE matrix; https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(matrix)
eigenvalues
print(eigenvalues)
print(eigenvectors)

matrix = np.array([[1, 4],
                   [2, 5]])
np.linalg.inv(matrix)          # invert of the matrix; http://www.mathwords.com/i/inverse_of_a_matrix.htm

# reshaping a matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [10, 11, 12]])
matrix.reshape(2, 6)  # reshaping the matrix into 2 rows 6 columns; only one parameter means 1 row; -1 as many as needed

# running a function on all elements of the matrix
add_100 = lambda i: i + 100                 # lambda function that adds 100 to something
vectorized_add_100 = np.vectorize(add_100)  # vectorized version of the add_100 function
print(vectorized_add_100(matrix))           # Applying function to all elements in matrix

print(matrix + 100)                         # using broadcasting - same result as above
# Broadcasting: to perform operations between arrays even if their dimensions are not the same

# Operations with matrices
matrix_a = np.array([[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 2]])
matrix_b = np.array([[1, 3, 1],
                     [1, 3, 1],
                     [1, 3, 8]])

np.add(matrix_a, matrix_b)              # OR matrix_a + matrix_b
np.subtract(matrix_a, matrix_b)         # OR matrix_a - matrix_b
np.dot(matrix_a, matrix_b)              # OR matrix_a @ matrix_b; multiplying matrices
matrix_a * matrix_b                     # multiply matrices element-wise


# *** Sparse matrix

matrix_zero = np.mat([[1, 0, 0], [0, 0, 0], [0, 6, 0]])
matrix_sparse = sparse.csr_matrix(matrix_zero)  # creating a compressed sparse row matrix (csr)

# Sparse matrices only store nonzero elements and assume all other values will be zero, leading to
# significant computational savings
# print(matrix_sparse)
# (0, 0)    1
# (2, 1)    6
# In compressed sparse row (CSR) matrices, (0, 0) and (2, 1) represent the (zero-indexed) indices of the non-zero values
# https://docs.scipy.org/doc/scipy/reference/sparse.html


# *** Generating random values

# np.random.seed(0)
print(np.random.random(3))          # three random floats between 0.0 and 1.0
print(np.random.randint(0, 11, 3))  # three random integers between 1 and 10
np.random.normal(0.0, 1.0, 3)     # three numbers from a normal distribution with mean 0.0 and standard deviation of 1.0
np.random.logistic(0.0, 1.0, 3)   # three numbers from a logistic distribution with mean 0.0 and scale of 1.0
np.random.uniform(1.0, 2.0, 3)    # three numbers greater than or equal to 1.0 and less than 2.0
