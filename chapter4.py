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
