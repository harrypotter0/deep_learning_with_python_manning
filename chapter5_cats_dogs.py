import os,shutil
# directory for larger dataset
original_dataset_dir = ''
# directory for smaller dataset
base_dir = ''
os.mkdir(base_dir)

# Directories for train,validation and test splits
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

#Directories with training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)

#Directories with training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_cats_dir)

#Directories with validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)

#Directories with training cat pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)

#Directories with testing cat pictures
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)

#Directories with testing dog pictures
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_cats_dir)

#############################################################
# copies the first 1000 cat images to train_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

# copies the first 1000 cat images to validation_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

# copies the first 1000 cat images to train_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

###############################################################
# copies the first 1000 cat images to train_cats_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

# copies the first 1000 cat images to validation_cats_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

# copies the first 1000 cat images to train_cats_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

print('total training cat images :', len(os.listdir(train_cats_dir)))
print('total training dog images :', len(os.listdir(train_dogs_dir)))
print('total validation cat images :', len(os.listdir(validation_cats_dir)))
print('total validation cat images :', len(os.listdir(validation_cats_dir)))
print('total test cat images :', len(os.listdir(test_cats_dir)))
print('total test cat images :', len(os.listdir(test_cats_dir)))

# Instantiating a small convent for dogs vs cats classification
from keras import layers 
from keras import models 

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu',
                        input_shape=((150, 150, 3))))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()
from keras import optimizers
model.compile(loss='binary_crossentopy',
            optimizer=optimizers.RMSprop(lr=le-4),
            metrics=['acc'])

## Preprocessing 
# Using ImageDataGenerator to read images from directories
from keras.preprocessing.image import ImageDataGenerator
#Rescales all images by 1/255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')
validation_generator=test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

# def generator():
#     i=0
#     while True:
#         i+=1
#         yield i

# for item in generator():
#     print(item)
#     if item>4:
#         break

for data_branch, labels_batch in train_generator:
    print('data batch shape :', data_batch.shape)
    print('labels batch shape :', labels_batch.shape)
    break

#fitting the model using a batch generator
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50
)

model.save('cats_and_dogs_small_1.h5')

#Displaying curves of loss and accuracy during training
import matplotlib.pyplot as plt 
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label=;Validation loss)
plt.title('Training and Validation loss')
plt.legend()
plt.show()

# Setting up a data augumentation via ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Displaying some randomly augumented training images
from keras.preprocessing import image
fnames = [os.path.join(train_cats_dir, fname)for 
        fname in os.listdir(train_cats_dir)]
img_path = fnames[3]
img=image.load_img(img_path, target_size=(150, 150))
x = image.img_to_array(img)
x = x.reshape((1,)+x.shape)
i=0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i+=1
    if i%4==0:
        break
plt.show()

# Define a new convnet that includes dropnet 
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', 
            optimizer=optimizers.RMSprop(lr=1e-4),
            metrics=['acc'])

# Training the convnet using data-augumentation generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40, 
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50
)

model.save('cats_and_dogs_small_2.h5')

## Using a pretrained convnet (VGGNet)

#1. Feature Exraction 
# Instantiating the VGG16 convolutional base
from keras.application import VGG16 

conv_base = VGG16(weights='imagenet',
                include_top=False,
                input_shape=(150 , 150, 3))

conv_base.summary()

# Extracting features using the pretrained convolutional base
import os 
import numpy as np 
from keras.preprocessing.image import ImageDataGenerator

base_dir = ''
train_dir = ''
validation_dir = ''
test_dir = ''

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20
def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150 150),
        batch_size=batch_size,
        class_mode='binary'
    )
    i=0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i*batch_size:(i+1)*batch_size] = features_batch
        labels[i*batch_size:(i+1)*batch_size] = labels_batch
        i+=1
        if i*batch_size>=sample_count:
            break
    return features, labels

train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

train_features = np.reshape(train_features, (200, 4*4*512))
validation_features = np.reshape(validation_features, (1000, 4*4*512))
test_features = np.reshape(test_features, (1000, 4*4*512))

# Defining and training the densely connected classifier
from keras import models 
from keras import layers 
from keras import optimizers 

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4*4*512))
model.add(layers.Dropout(0,5))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
            loss='binary_crossentropy',
            metrics=['acc'])
history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features,validation_labels)
                    )

# Plotting the results
import matplotlib.pyplot as plt 
acc = history,history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation_loss')
plt.title('Training and validation accuracy')
plt.legend()

plt.show()

# adding a densely connected classifier on top of the convolutional base
from keras import models 
from keras import layers

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

print("this is the no of trainable weights before freezing the conv base", len(model.trainable_weights))
conv_base.trainable = False
print("this is the no of trainable weights after freezing the conv base", len(model.trainable_weights))

# training the model end to end with a frozen convolutional base
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

model.compile(loss='binary_crossentropy',
            optimizer=optimizers.RMSprop(lr=2e-5),
            metrics=['acc'])
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50
)

## 2.Fine Tuning
# 1. add your custom n/w on top of an already-trained base n/w 
# 2. freeze the base n/w
# 3. train the part you added.
# 4. unfreeze some layers in the same n/w 
# 5. jointly train both these layers and the part you added. 

conv_base.summary()

# Freeze all layers up to a specific one 
conv_base.trainable = True 
set_trainable = False 
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True 
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

# Fine tuning the model
model.compile(loss='binary_crossentropy',
            optimizer=optmizers.RMSprop(lr=1e-5),
            metrics=['acc'])
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50
)

# Smoothing the plots
def smooth_curve(points, factor=0.8):
    smoothed_points =[]
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous*factor+point*(1-factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

plt.plot(epochs, smooth_curve(acc), 'bo', label='Smoothed training acc')
plt.plot(epochs, smooth_curve(val_acc), 'b', label='Smooth validation acc')
plt.legend()

plt.figure()
plt.plot(epochs, smooth_curve(loss), 'bo', label='Smooth training loss')
plt.plot(epochs, smooth_curve(loss), 'b', label='Smooth validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)
test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test_acc: ', test_acc)

# Visualizing intermediate activations
from keras.models import load_model
model = load_model('cats_and_dogs_small_2.h5')
model.summary()

# Preprocessing a single image
img_path = ''
from keras.preprocessing import image
import numpy as np 
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /=255.

print(tensor.shape)

# Displaying the test picture
import matplotlib.pylplot as plt 
plt.imshow(img_tensor[0])
plt.show()

# Instantiating a model from an input tensor and a list of output tensors
from keras import models 
layers_outputs = [layer.ouptut for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

# Running the model in predict mode
activations= activation_model.predict(img.tensor)

# Visualizing the fourth channel 
import matplotlib.pyplot as plt 
plt.matshow(first_layer_activation[0, :, :, 4], cmap='virdis')

# Visualizing every channel in every intermediate activation
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)
images_per_row = 16
for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    n_cols = n_features//images_per_row
    display_grid = np.zeros((size*n_cols, images_per_row*size))
    for cols in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, 
                                            :,:,
                                            col*images_per_row+row]
            channel_image -=channel_image.mean()
            channel_image /= channel_image.std()
            channel_image*=64
            channel_image+=128
            channel_image= np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col*size : (col+1)*size,
                        row*size : (row+1)*size] = channel_image
    scale =1./size
    plt.figure(figsize=(scale*display_grid.shape[1],
                        scale*display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')

## Visualizing convnet filters
# Defining the loss tensor for filter visualization
from keras.applications import VGG16
from keras import backend as K 

model = VGG16(weights='imagenet',
                include_top=False)
layer_name='block3_conv1'
filter_index = 0
layer_output = model.get_layer(layer_name).output
loss = K.mean(layer_output[:,:,:,filter_index])

# Obtaining the gradient of the loss with regard to the input 
grads = K.gradients(loss, model.input)[0]

# Gradient-normalization trick
grads/=(K.sqrt(K.mean(K.square(grade)))+ 1e-5)

# Fetching numpy output values given numpy input values
iterate = K.function([model.input], [loss, grads])
import numpy as np 
loss_value, grads_value = iterate(np.zeros((1, 150, 150, 3)))

# Loss maximization via stochastic gradient descent
input_img_data = np.random.random((1, 150, 150, 3))*20 + 128
step =1
for i in range(40):
    loss_value, grads_value = iterate([input_img_data])
    input_img_data += grads_value *step

# utility function to convert a tensor into a valid image
def deprocess_image(x):
    # normalizes the tensor: centers on 0, ensures that std is 0.1
    x-=x.mean()
    x/=(x.std()+1e-5)
    x*=0.1
    # clips to [0,1]
    x+=0.5
    x= np.clip(x,0,1)
    # convert to RGB array
    x*=255
    x= np.clip(x,0,255).astype('uint8')
    return x

# function to generate filter visualization
def generate_pattern(layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output
    loss= K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads/=(K.sqrt(K.mean(K.square(grads)))+1e-5)
    iterate = K.function([model.input], [loss, grads])
    input_img_data = np.random.random((1, size, size, 3))*20 + 128 
    step =1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value *step

    img = input_img_data[0]
    return deprocess_image[img] 

plt.imshow(generate_pattern('block3_conv1', 0))

# Generating a grid of all filter response patterns in a layer
layer_name = 'block_conv1'
size= 64
margin =5 
results = np.zeros((8*size+7*margin, 8*size+7*margin, 3))
for i in range(8):
    for j in range(8):
        filter_img = generate_pattern(layer_name, i+(j*8), size=size)
        horizontal_start = i*size+i*margin
        horizontal_end = horizontal_start + size 
        vertical_start = j*size + j*margin 
        vertical_end = vertical_start + size
        results[horizontal_start: horizontal_end,
                vertical_start: vertical_end, :] = filter_img
plt.figure(figsize=(20, 20))
plt.imshow(results)

# Visualizing heatmaps of class activation
# Loading the VGG16 n/w with pretrained weights
from keras.applications.vgg16 import VGG16
model = VGG16(weights= 'imagenet')

# Preprocessing an input image for VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np 
img_path = ''
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = model.predict(x)
print('Predicted', decode_predictions(preds, top=3)[0])
np.argmax(preds[0])

# setting up GRAD-CAM algorithm
african_elephant_output = model.output[:, 386]
last_conv_layer = model.get_layer('block5_conv3')
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input],[pooled_grads, last_conv_layer.output[0]])
pooled_grads_value conv_layer_output_value = iterate([x])
for i in range(512):
    conv_layer_output_value[:,:,i] *= pooled_grads_value[i]
heatmap = np.mean(conv_layer_output_value, axis=-1)

# Heatmap post-processing 
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
import cv2 
img = cv2.imreading(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255*heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap*0.4+img 
cv2.imwrite('',superimposed_img)










