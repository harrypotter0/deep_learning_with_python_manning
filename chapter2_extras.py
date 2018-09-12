# Scalars
import numpy as np 
x = np.array(12)
print(x)
print(x.ndim)

# Vectors
x = np.array([12, 3, 6, 16])
print(x)
print(x.ndim)

# Matrices
x = np.array([  [5, 78, 2, 34, 0],
                [6, 79, 3, 35, 1],
                [7, 80, 4, 36, 2]])
print(x.ndim)

from keras.datasets import mnist
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()
train_images = train_images[:,7:-7,7:-7]
digit = train_images[4]
import matplotlib.pyplot as plt 
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()


# Vector data— 2D tensors of shape (samples, features)
# Timeseries data or sequence data— 3D tensors of shape (samples, timesteps,
# features)
# Images— 4D tensors of shape (samples, height, width, channels) or (samples,
# channels, height, width)
# Video— 5D tensors of shape (samples, frames, height, width, channels) or
# (samples, frames, channels, height, width)

# 1st axis --> samples 
# 2nd axis --> features
# 3rd -> timesteps

## Relu operations
def naive_relu(x):
    assert len(x.shape)==2
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j],0)
    return x

def naive_add(x, y):
    assert len(x.shape)==2
    assert x.shape == y.shape
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j]+=y[i,j]
    return x

# import numpy as np 
# z  = x+y
# z = np.maximum(z,0.)

## Broadcasting
def naive_add_matrix_and_vector(x,y):
    assert len(x.shape)==2
    assert len(y.shape)==1
    assert x.shape[1]==y.shape[0]
    x= x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j]+=y[j]
    return x

import numpy as np 
x = np.random.random((64, 3,32,10))
y = np.random.random((32,10))
z= np.maximum(x,y)

## TENSOR DOT
import numpy as np 
x = np.dot(x,y)

def naive_vector_dot(x,y):
    assert len(x.shape)==1
    assert len(y.shape)==1
    assert x.shape[0] == y.shape[0]
    z= 0 
    for i in range(x.shape[0]):
        z+=x[i]*y[i]
    return z

def naive_matrix_vector_dot(x,y):
    assert len(x.shape)==2
    assert len(y.shape)==1
    assert x.shape[1]==y.shape[0]
    z= np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i] += x[i,j]*y[i]
    return z

def naive_matrix_vector_dot(x,y):
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        z[i] = naive_vector_dot(x[i,:],y)
    return z

def naive_matrix_dot(x,y):
    assert len(x.shape)==2
    assert len(y.shape)==2
    assert x.shape[1] == y.shape[0]
    x= np.zeros((x.shape[0],y.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            row_x = x[i,:]
            column_y = y[:,j]
            z[i,j] = naive_vector_dot(row_x, column_y)
    return z

## TENSOR RESHAPING
x = np.array([  [0., 1.],
                [2., 3.],
                [4., 5.]])
print(x.shape)
x = x.reshape((6,1))
print(x)
x = np.zeros((300, 20))
x = np.transpose(x)
print(x.shape)

