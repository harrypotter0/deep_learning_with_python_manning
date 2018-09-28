#Hold Out Validation
num_validation_samples =10000
np.random.shuffle(data)

validation_data = data[:num_validation_samples]
data = data[num_validation_samples:]

training_data = data[:]

model = get_model()
model.train(training_data)
validation_score = model.evaluate(validation_data)

#At this point you can tune your model,
# retrain it,evaluate it and train it again..
# ..

model = get_model()
model.train(np.concatenate([training_data, validation_data]))
test_score = model.evaluate(test_data)

#K-fold cross validation
k= 4
num_validation_samples = len(data)//k 
np.random.shuffle(data)

validation_stores = []
for fold in range(k):
    validation_data = data[num_validation_samples*fold:num_validation_samples*(fold+1)]
    training_data = data[:num_validation_samples*fold]
    data[num_validation_samples*(fold+1):]
    model = get_model()
    model.train(training_data)
    validation_score = model.evaluate(validation_data)
    validation_scores.append(validation_score)

validation_score = np.average(validation_scores)

model = get_model()
model.train(data)
test_score = model.evaluate(test_data)

x-=x.mean(axis=0)
x/=x.std(axis=0)

# Original Model 
from keras import models 
from keras import layers 

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#Version of the model with lower capacity
model = models.Sequential()
model.add(layers.Dense(4, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Version of the model with higher capacity
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# L1 regularization—The cost added is proportional to the absolute value of the
# weight coefficients (the L1 norm of the weights).
#  L2 regularization—The cost added is proportional to the square of the value of the
# weight coefficients (the L2 norm of the weights).
from keras import regularizers 
model = models.Sequential()
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
                        activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
                        activation='relu', input_shape=(10000,)))
model.add(layers.Dense(1, activation='sigmoid'))       

#Different weight regularizers available in Keras 
from keras import regularizers
regularizers.l1(0.001)
regularizers.l1_l2(0.001, 0.001)

#Dropout
layer_output *= np.random.randint(0, high=2, size=layer_output.shape)
layer_output *=0.5

layer_output *= np.random.randint(0, high=2, size=layer_output.shape)
layer_output /=0.5

model1.add(layers.Dropout(0.5))

# Adding dropout to the IMDB network 
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

# These are the most common ways to prevent overfitting in neural networks:
#  Get more training data.
#  Reduce the capacity of the network.
#  Add weight regularization.
#  Add dropout.

# Binary classification sigmoid binary_crossentropy
# Multiclass, single-label classification softmax categorical_crossentropy
# Multiclass, multilabel classification sigmoid binary_crossentropy
# Regression to arbitrary values None mse
# Regression to values between 0 and 1 sigmoid mse or binary_crossentropy

# Scaling up: developing a model that overfits
# To figure out how big a model you’ll need, you must develop a model that overfits.
# This is fairly easy:
# 1# Add layers.
# 2# Make the layers bigger.
# 3# Train for more epochs.
# Always monitor the training loss and validation loss, as well as the training and valida-
# tion values for any metrics you care about. When you see that the model’s perfor-
# mance on the validation data begins to degrade, you’ve achieved overfitting.
# The next stage is to start regularizing and tuning the model, to get as close as pos-
# sible to the ideal model that neither underfits nor overfits.

# Regularizing your model and tuning your hyperparameters
#  Add dropout.
#  Try different architectures: add or remove layers.
#  Add L1 and/or L2 regularization.
# Try different hyperparameters (such as the number of units per layer or the
# learning rate of the optimizer) to find the optimal configuration.
#  Optionally, iterate on feature engineering: add new features, or remove fea-
# tures that don’t seem to be informative.
