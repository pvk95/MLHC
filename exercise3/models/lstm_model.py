from tensorflow.keras.layers import Input, Convolution1D, MaxPool1D, Dropout, Dense, GlobalMaxPool1D, LSTM, Reshape, CuDNNLSTM, Bidirectional
from tensorflow.keras import activations, models, optimizers, losses
from .model import Model


class LSTM_Model(Model):
    def __init__(self, input_shape=(186, 1), outputs=1, epochs=1000, summary=False, verbose=1, hidden=32, dense=64):
        super().__init__(input_shape, outputs, epochs, summary, verbose)
        self.hidden = hidden
        self.dense = dense
        self.model = self.getModel()
    
    def fit(self, X, y):
        self.model = self.getModel()
        return super().fit(X, y)
    
    def getModel(self):
        inp = Input(shape=self.input_shape)
        lstm = Bidirectional(CuDNNLSTM(self.hidden))(inp)
        #lstm = Bidirectional(LSTM(self.hidden))(inp)
        dense_1 = Dense(self.dense, activation=activations.relu)(lstm)
        dense_1 = Dense(self.dense, activation=activations.relu)(dense_1)

        if self.outputs == 1:
            dense_1 = Dense(self.outputs, activation=activations.sigmoid)(dense_1)
        else:
            dense_1 = Dense(self.outputs, activation=activations.softmax)(dense_1)

        model = models.Model(inputs=inp, outputs=dense_1)
        opt = optimizers.Adam(0.001)

        if self.outputs == 1:
            model.compile(optimizer=opt, loss=losses.binary_crossentropy, metrics=['acc'])
        else:
            model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])

        if self.summary:
            model.summary()
        return model