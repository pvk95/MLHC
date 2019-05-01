import pandas as pd
import numpy as np
import models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# Load mitbih data
mitbih_train = pd.read_csv('exercise_data/heartbeat/mitbih_train.csv', header=None)

X = mitbih_train[list(range(186))].values[..., np.newaxis]
Y = mitbih_train[187]
X, Y = shuffle(X, Y, random_state=42)
Y_binary = to_categorical(Y)

# Train model and save it
rnn_model = models.LSTM_Model(outputs=5)
rnn_model.fit(X, Y_binary)
rnn_model.model.save('transfer_learning.h5')

# load ptbdb data
ptbdb_1 = pd.read_csv("exercise_data/heartbeat/ptbdb_normal.csv", header=None)
ptbdb_2 = pd.read_csv("exercise_data/heartbeat/ptbdb_abnormal.csv", header=None)
ptbdb = pd.concat([ptbdb_1, ptbdb_2])
X = ptbdb[list(range(186))].values[..., np.newaxis]
Y = ptbdb[187]
X, Y = shuffle(X, Y, random_state=42)

# train method for later use
def train(model, X, Y):
    early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
    redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=1)
    callbacks_list = [early, redonplat]  # early

    model.fit(X, Y, epochs=1000, verbose=1, callbacks=callbacks_list, validation_split=0.1)


# Freeze everything except the last layer and train
model = load_model('transfer_learning.h5')
out = Dense(1, name='ouput')(model.layers[-2].output)
model2 = Model(model.input, out)

for x in range(4):
    model2.layers[x].trainable = False

model2.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['acc'])
train(model2, X, Y)

# Don't freeze anything
model = load_model('transfer_learning.h5')
out = Dense(1, name='ouput')(model.layers[-2].output)
model2 = Model(model.input, out)
model2.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['acc'])
train(model2, X, Y)

# First freeze base model, then train whole model
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=0.1)

model = load_model('transfer_learning.h5')
out = Dense(1, name='ouput')(model.layers[-2].output)
model2 = Model(model.input, out)

for x in range(4):
    model2.layers[x].trainable = False

model2.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['acc'])

early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=1)
callbacks_list = [early, redonplat]  # early

model2.fit(X_train, Y_train, epochs=1000, verbose=1, callbacks=callbacks_list, validation_data=(X_test, Y_test))

for x in range(4):
    model2.layers[x].trainable = True
model2.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['acc'])
model2.summary()
model2.fit(X_train, Y_train, epochs=1000, verbose=1, callbacks=callbacks_list, validation_data=(X_test, Y_test))