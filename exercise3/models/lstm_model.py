from tensorflow.keras.layers import Input, Convolution1D, MaxPool1D, Dropout, Dense, GlobalMaxPool1D, LSTM, Reshape
from tensorflow.keras import activations, models, optimizers, losses
from .model import Model


class LSTM_Model(Model):
    def __init__(self, name='Model', input_shape=(186, 1), outputs=1, epochs=1000, summary=False):
        super().__init__(name, input_shape, outputs, epochs, summary)
        self.model = self.getModel()
    
    def getModel(self):
        inp = Input(shape=self.input_shape)
        #lstm = CuDNNLSTM(100)(inp)
        lstm = LSTM(100)(inp)
        lstm = Reshape((100, 1))(lstm)

        img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(lstm)
        img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(img_1)
        img_1 = MaxPool1D(pool_size=2)(img_1)
        img_1 = Dropout(rate=0.1)(img_1)
        img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
        img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
        img_1 = MaxPool1D(pool_size=2)(img_1)
        img_1 = Dropout(rate=0.1)(img_1)
        img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
        img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
        img_1 = MaxPool1D(pool_size=2)(img_1)
        img_1 = Dropout(rate=0.1)(img_1)
        img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
        img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
        img_1 = GlobalMaxPool1D()(img_1)
        img_1 = Dropout(rate=0.2)(img_1)

        dense_1 = Dense(64, activation=activations.relu, name="dense_1")(img_1)
        dense_1 = Dense(64, activation=activations.relu, name="dense_2")(dense_1)
        dense_1 = Dense(self.outputs, activation=activations.sigmoid, name="dense_3_ptbdb")(dense_1)

        model = models.Model(inputs=inp, outputs=dense_1)
        opt = optimizers.Adam(0.001)

        model.compile(optimizer=opt, loss=losses.binary_crossentropy, metrics=['acc'])
        if self.summary:
            model.summary()
        return model