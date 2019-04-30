from tensorflow.keras.layers import Input, Convolution1D, MaxPool1D, Dropout, Dense, GlobalMaxPool1D, ReLU, Add, Flatten
from tensorflow.keras import activations, models, optimizers, losses
from .model import Model

class Residual_CNN(Model):
    def __init__(self, name='Model', input_shape=(186, 1), outputs=1, epochs=1000, summary=False):
        super().__init__(name, input_shape, outputs, epochs, summary)
        self.model = self.getModel()
    
    def getModel(self):
        inp = Input(self.input_shape)
        res = Convolution1D(32, kernel_size=5, activation=activations.linear)(inp)

        for _ in range(5):
            res = self.residual_block(res)

        flatten = Flatten()(res)
        dense1 = Dense(32, activation='relu')(flatten)

        if self.outputs == 1:
            output = Dense(self.outputs, activation='relu')(dense1)
        else:
            output = Dense(self.outputs, activation='softmax')(dense1)

        model = models.Model(inputs=inp, outputs=output)
        opt = optimizers.Adam(0.001)

        model.compile(optimizer=opt, loss=losses.binary_crossentropy, metrics=['acc'])
        if self.summary:
            model.summary()
        return model

    def residual_block(self, conv):
        conv1 = Convolution1D(32, kernel_size=5, activation=activations.relu, padding='same')(conv)
        conv2 = Convolution1D(32, kernel_size=5, activation=activations.linear, padding='same')(conv1)
        add =  Add()([conv2, conv])
        relu = ReLU()(add)
        pool = MaxPool1D(5, strides=2)(relu)
        return pool

