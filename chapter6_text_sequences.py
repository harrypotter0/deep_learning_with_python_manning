# word-level one-hot encoding 
import numpy as np 
samples = ['the cat sat on the mat.','The dog ate my homework.']
token_index = {}
for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index)+1
max_length =10
results = np.zeros(shape=(
    len(samples),
    max_length,
    max(token_index.values())+1
))
for i,sample in enumerate(samples):
    for j,word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] =1.

# character-level one hot encoding 
import string 
samples = ['the cat sat on the mat.', 'the dof ate my ass.']
characters = string.printable
token_index = dict(zip(range(1,len(characters))+1), characters))
max_length = 50 
results = np.zeros((len(samples)), max_length, max(token_index.keys())+1))
for i, sample in enumerate(samples):
    for j,character in enumerate(sample):
        index= token_index.get(character)
        results[i, j, index]=1

# Using keras for word level one hot encoding 

from keras.preprocessing.text import Tokenizer 
samples = ['the cat sat on the floor.', 'the dog ate my ass.']
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples)
sequences = tokenizer.texts_to_sequences(samples)
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
word_index = tokenizer.word_index
print("Found %s unique tokens",%len(word_index))

# word-level encoding with hashing trick 
samples = []
dimensionality = 1000
max_length =10 
results = np.zeros((len(samples), max_length, dimensionality))
for i, sample in enumerate(samples):
    for j,word in list(enumerate(sample.split()))[:max_length]:
        index = abs(hash(word))%dimensionality
        results[i, j, index]=1.

# Word Embedding using embedding layer
# Instantiating an embedding layer
from keras.layers import Embedding 
embedding_layer = Embedding(1000,64)

# loading the imdb data for use with an embedding layer
from keras.datasets import imdb 
from keras import preprocessing 
max_features = 10000
maxlen = 20 
(x_train, y_train),(x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# using an embedding layer and classifier on the IMDB data
from keras.models import Sequential
from keras.layers import Flatten, Dense 
model = Sequential()
model.add(Embedding(10000, 8, input_length=max_len))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['acc']
)
model.summary()
history = model.fit(
    x_train,y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

# Using pretrained word embedding 
# processing the labels of the raw IMDB data 
import os 
imdb_dir = ''
train_dir = ''
labels = []
texts = []
for lebel_type in ['neg','pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type =='neg':
                labels.append(0)
            else:
                labels.append(1)

# tokenize the text of the raw IMDB data 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

maxlen =100
training_samples = 200
validation_samples = 10000
max_words = 10000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found to unique tokens '%len(word_index))
data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)
print('Shape of data tensor',data.shape)
print('Shape of label tensor', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples+validation_samples]
y_val = labels[training_samples: training_samples+validation_samples]


# Parsing the Glove word-embedding file 
glove_dir = ''
embedding_index = []
f = open(os.path.join(glove_dir, 'glove-6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index(word) = coefs
f.close()
print('Found %s word vectors' %len(embedding_index))

# preparing the glove word-embeddings matrix
embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i<max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None :
            embedding_matrix[i]  = embedding_vector
    
# model defintion
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# loading pretrained word  embeddings into the embedding layer 
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

# training and evaluating the model
model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['acc']
)
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(x_val, y_val)
)
model.save_weights('pre_trained_glove_model.h5')

# Plotting the results
import matplotlib.pyplot as plt 
acc = history.history['acc']
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
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# Training the same model without pretrained word embeddings 
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense 

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['acc']
)
history= model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(x_val, y_val)
)

# Tokenizing the data of the test set 
test_dir = os.path.join(imdb_dir, 'test')
labels = []
texts = []

for label_type in ['neg','pos']:
    dir_name = os.path.join(test_dir, label_type)
    for fname in sorted(os.listdir(dir_name)):
        if fname[-4:] =='.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type=='neg':
                labels.append[0]
            else:
                labels.append[1]

sequences = tokenizer.texts_to_sequences[texts]
x_test = pad_sequences(sequences, maxlen=maxlen)
y_test = np.asarray(labels)

# evaluating the first model
model.load_weights('pre-trained_glove_model.h5')
model.evaluate(x_test, y_test)

# Recurrent Neural Networks
# Pseudocode RNN
state_t =0 
for input_t in input_sequence:
    output_t = f(input_t, state_t)
    state_t = output_t

# more detailed rnn
state_t = 0
for input_t in input_sequence:
    output_t = activation(dot(W, input_t)+dot(U, state_t)+b)
    state_t = output_t

# numpy implementation of a simple RNN 
import numpy as np 
timesteps = 100
input_features = 32 
output_features = 64
inputs = np.random.random((timesteps, input_features))
state_t = np.zeros((output_features, ))
W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features))
successive_outputs = []
for input_t in inputs:
    output_t = np.tanh(np.dot(W, input_t)+np.dot(U, state_t)+b)
    successive_outputs.append(output_t)
    state_t = output_t
final_output_sequence = np.concatenate(successive_outputs, axis=0)

# Recurrent layer in keras
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN
model =Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32))
model.summary()

# Preparing the IMDB Data 
from keras.datasets import imdb 
from keras.preprocessing import sequence 
max_features = 10000
maxlen = 500
batch_size = 32 
print("Loading data")
(input_train, y_train),(input_test, y_test) = imdb.load_data(num_words=max_features)
print(len(input_train), "train sequences")
print(len(input_test), "test sequences")

print("pad sequences samples x time")
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print("input_train shape", input_train.shape)
print("input_test shape", input_test.shape)

# Training the model with Embedding and SimpleRNN layers
from keras.layers import Dense 
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))
model.compile(
    optimizer='rmsprop',
    loss='binarycrossentropy',
    metrics=['acc'])
history = model.fit(
    input_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2
)

# Plotting results
import matplotlib.pyplot as plt 
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='validation acc')
plt.title('Training and validation accuracy ')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# Understanding the LSTM and GRU layers
# Pseudocode LSTM
output_t = activation(dot(state_t, Uo)+dot(input_t, Wo)+dot(C_t, Vo)+ bo)
i_t = activation(dot(state_t, Ui)+dot(input_t, Wl)+bi)
f_t = activation(dot(state_t, Uf)+dot(input_t, Wf)+bf)
k_t = activation(dot(state_t, Uk)+dot(input_t, Wk)+bk)
c_t+1 = i_t+k_t+c_t*f_t




















