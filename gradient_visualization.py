import numpy as np
import matplotlib.pyplot as plt


# generate 2D meshgrid
nx, ny = (100, 100)
x = np.linspace(0, 10, nx)
y = np.linspace(0, 10, ny)

xv, yv = np.meshgrid(x, y)
# print(xv)


# define a function to plot
def f(x, y):
    return x * (y**2)


# calculate z value for each x, y point
z = f(xv, yv)
print(z.shape)


# make a colour plot to display the data
plt.figure(figsize=(14, 12))
plt.pcolor(xv, yv, z)
plt.title('2D Colour Plot of f(x,y) = xy**2')
plt.colorbar()
plt.show()


# generate new meshgrid for the gradient
nx, ny = (10, 10)
x = np.linspace(0, 10, nx)
y = np.linspace(0, 10, ny)

xg, yg = np.meshgrid(x, y)

# calculate the gradient of f(x,y)
gy, gx = np.gradient(f(xg, yg))  # np deals with rows first, and y is gonna be the the row actually

# make a colour plot to display the data
plt.figure(figsize=(14, 12))
plt.pcolor(xv, yv, z)
plt.title('Gradient of f(x,y) = xy**2')
plt.colorbar()
plt.quiver(xg, yg, gx, gy, scale=1000, color='w')  #
plt.show()


# check result, calculate the gradient of f(x,y) = xy**2
def ddx(x, y):
    return y**2


def ddy(x, y):
    return 2 * x * y


gx = ddx(xg, yg)
gy = ddy(xg, yg)

# checking if result is the same
plt.figure(figsize=(14, 12))
plt.pcolor(xv, yv, z)
plt.title('Checking result of Gradient of f(x,y) = xy**2')
plt.colorbar()
plt.quiver(xg, yg, gx, gy, scale=1000, color='w')  #
plt.show()
