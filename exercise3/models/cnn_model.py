from tensorflow.keras.layers import Input, Convolution1D, MaxPool1D, Dropout, Dense, GlobalMaxPool1D
from tensorflow.keras import activations, models, optimizers, losses
from .model import Model


class CNN_Model(Model):
    def __init__(self, name='Model', input_shape=(186, 1), outputs=1, epochs=1000, summary=False, verbose=1, 
                conv1_size=16, conv2_size=32, conv3_size=256, dense_size=64):
        super().__init__(name, input_shape, outputs, epochs, summary, verbose)
        self.conv1_size = conv1_size
        self.conv2_size = conv2_size
        self.conv3_size = conv3_size
        self.dense_size = dense_size
        self.model = self.getModel()
    
    def fit(self, X, y):
        self.model = self.getModel()
        return super().fit(X, y)
    
    def get_params(self, deep=True):
        return {
            'conv1_size': self.conv1_size,
            'conv2_size': self.conv2_size,
            'conv3_size': self.conv3_size,
            'dense_size': self.dense_size
        }
    
    def getModel(self):
        inp = Input(shape=self.input_shape)
        img_1 = Convolution1D(self.conv1_size, kernel_size=5, activation=activations.relu, padding="valid")(inp)
        img_1 = Convolution1D(self.conv1_size, kernel_size=5, activation=activations.relu, padding="valid")(img_1)
        img_1 = MaxPool1D(pool_size=2)(img_1)
        img_1 = Dropout(rate=0.1)(img_1)
        img_1 = Convolution1D(self.conv2_size, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
        img_1 = Convolution1D(self.conv2_size, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
        img_1 = MaxPool1D(pool_size=2)(img_1)
        img_1 = Dropout(rate=0.1)(img_1)
        img_1 = Convolution1D(self.conv2_size, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
        img_1 = Convolution1D(self.conv2_size, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
        img_1 = MaxPool1D(pool_size=2)(img_1)
        img_1 = Dropout(rate=0.1)(img_1)
        img_1 = Convolution1D(self.conv3_size, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
        img_1 = Convolution1D(self.conv3_size, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
        img_1 = GlobalMaxPool1D()(img_1)
        img_1 = Dropout(rate=0.2)(img_1)

        dense_1 = Dense(self.dense_size, activation=activations.relu, name="dense_1")(img_1)
        dense_1 = Dense(self.dense_size, activation=activations.relu, name="dense_2")(dense_1)
        dense_1 = Dense(self.outputs, activation=activations.sigmoid, name="dense_3_ptbdb")(dense_1)

        model = models.Model(inputs=inp, outputs=dense_1)
        opt = optimizers.Adam(0.001)

        model.compile(optimizer=opt, loss=losses.binary_crossentropy, metrics=['acc'])
        if self.summary:
            model.summary()
        return model