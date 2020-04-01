import numpy as np
from scipy import sparse


#  Vector
vector_row = np.array([1, 2, 3])
vector_column = np.array([[1], [2], [3]])

print("VECTOR")
print(vector_row[0])     # 1
print(vector_column[2])  # [3]
vector_row[:]     # all elements of the vector; [1 2 3]
vector_row[1:]    # all elements after the first; [2 3]
vector_row[:1]    # all elements up to and including the first; [1]
vector_row[-1]    # last element; 3


#  Matrix
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

add_100 = lambda i: i + 100                 # lambda function that adds 100 to something
vectorized_add_100 = np.vectorize(add_100)  # vectorized version of the add_100 function
print(vectorized_add_100(matrix))           # Applying function to all elements in matrix

print(matrix + 100)                         # using broadcasting - same result as above
# Broadcasting: to perform operations between arrays even if their dimensions are not the same


#  Sparse matrix
matrix_zero = np.mat([[1, 0, 0], [0, 0, 0], [0, 6, 0]])
matrix_sparse = sparse.csr_matrix(matrix_zero)  # creating a compressed sparse row matrix (csr)

# Sparse matrices only store nonzero elements and assume all other values will be zero, leading to
# significant computational savings
# print(matrix_sparse)
# (0, 0)    1
# (2, 1)    6
# In compressed sparse row (CSR) matrices, (0, 0) and (2, 1) represent the (zero-indexed) indices of the non-zero values
# https://docs.scipy.org/doc/scipy/reference/sparse.html

