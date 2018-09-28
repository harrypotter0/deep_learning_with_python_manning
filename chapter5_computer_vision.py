#convnets
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3,3), activation='relu' ))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))

model.summary()

# Adding a dense layer on top of convnet
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

#Training the convnet on MNIST images
from keras.datasets import mnist 
from keras.utils import to_categorical

(train_images, train_labels),(test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32')/255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=64)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)

# convnets vs dense layers
# convnets learn local patterns, dense go for global patterns
# convnets are translation invariant
# Layer (type)
# Output Shape
# Param #
# ================================================================
# conv2d_1 (Conv2D)
# (None, 26, 26, 32)
# 320
# ________________________________________________________________
# maxpooling2d_1 (MaxPooling2D)
# (None, 13, 13, 32)
# 0
# ________________________________________________________________
# conv2d_2 (Conv2D)
# (None, 11, 11, 64)
# 18496
# ________________________________________________________________
# maxpooling2d_2 (MaxPooling2D)
# (None, 5, 5, 64)
# 0
# ________________________________________________________________
# conv2d_3 (Conv2D)
# (None, 3, 3, 64)
# 36928
# ________________________________________________________________
# flatten_1 (Flatten)
# (None, 576)
# 0
# ________________________________________________________________
# dense_1 (Dense)
# (None, 64)
# 36928
# ________________________________________________________________
# dense_2 (Dense)
# (None, 10)
# 650
# ================================================================
# Total params: 93,322
# Trainable params: 93,322
# Non-trainable params: 0


