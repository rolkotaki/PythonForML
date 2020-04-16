import numpy as np
import matplotlib.pyplot as plt


A = np.arange(9) - 3   # vector; [-3 -2 -1  0  1  2  3  4  5]
B = A.reshape((3, 3))  # matrix

# Euclidean norm (L2) - default
print(np.linalg.norm(A))
print(np.linalg.norm(B))
print(np.linalg.norm(B, 'fro'))  # Frogenius norm is the same as L2 norm

# the max norm
print(np.linalg.norm(A, np.inf))
print(np.linalg.norm(B, np.inf))

# vector normalization - to produce a unit vector
norm = np.linalg.norm(A)
A_unit = A / norm
print(A_unit)
# the magnitude of a unit vector is equal to 1
print(np.linalg.norm(A_unit))

# find the eigenvalues and eigenvectors
A = np.diag(np.arange(1, 4))
print(A)
eigenvalues, eigenvectors = np.linalg.eig(A)
print(eigenvalues)
print(eigenvectors)
print(eigenvalues[1])
print(eigenvectors[:, 1])

# verify eigendecomposition
print(np.diag(eigenvalues))  # with eigenvalues on the diagonal
print(np.linalg.inv(eigenvectors))  # the same by coincident
matrix = np.matmul(np.diag(eigenvalues), np.linalg.inv(eigenvectors))
output = np.matmul(eigenvectors, matrix).astype(np.int)
print(output)


# plot the eigenvectors
origin = [0, 0, 0]
fig = plt.figure(figsize=(18, 10))
fig.suptitle('Effects of Eigenvalues and Eigenvectors')
ax1 = fig.add_subplot(121, projection='3d')
ax1.quiver(origin, origin, origin, eigenvectors[0, :], eigenvectors[1, :], eigenvectors[2, :], color='k')
ax1.set_xlim([-3, 3])
ax1.set_ylim([-3, 3])
ax1.set_zlim([-3, 3])
ax1.set_xlabel('X axis')
ax1.set_ylabel('Y axis')
ax1.set_zlabel('Z axis')
ax1.view_init(15, 30)
ax1.set_title('Before Multiplication')

# multiply original matrix by eigenvectors
nuw_eig = np.matmul(A, eigenvectors)
ax2 = plt.subplot(122, projection='3d')
ax2.quiver(origin, origin, origin, nuw_eig[0, :], nuw_eig[1, :], nuw_eig[2, :], color='k')
# add eigenvalues to the plot
ax2.plot(eigenvalues[0]*eigenvectors[0], eigenvalues[1]*eigenvectors[1], eigenvalues[2]*eigenvectors[2], 'rX')
ax2.set_xlim([-3, 3])
ax2.set_ylim([-3, 3])
ax2.set_zlim([-3, 3])
ax2.set_xlabel('X axis')
ax2.set_ylabel('Y axis')
ax2.set_zlabel('Z axis')
ax2.view_init(15, 30)
ax2.set_title('After Multiplication')

plt.show()
