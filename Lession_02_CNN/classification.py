from keras.datasets import cifar10
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from keras.layers import Input ,Conv2D, Activation, MaxPool2D, Dropout
from keras.layers import Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.callbacks import CSVLogger

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
name_labels = cifar10
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# fig, axs  = plt.subplots(2,2)
# axs  = axs.flatten()
# for n, ax in enumerate(axs):
#     ax.imshow(x_train[n])
# plt.show()
num_classes = 10

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

def create_model():
    model_input = Input(shape=(32, 32, 3))
    x = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(model_input)
    x = Activation("relu")(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Dropout(rate=0.25)(x)

    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same")(model_input)
    x = Activation("relu")(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Dropout(rate=0.25)(x)

    x = Flatten()(x)
    x = Dense(units=512)(x)
    x = Activation("relu")(x)
    x = Dropout(rate=0.5)(x)

    x = Dense(units=num_classes)(x)
    model_output = Activation("softmax")(x)
    model = Model(inputs=model_input, outputs=model_output)
    return model

optimizer = Adam(learning_rate=0.001)

model = create_model()
model.compile(optimizer=optimizer, loss=categorical_crossentropy, metrics=["accuracy"])
callback = [CSVLogger(filename="log.csv", separator=",", append=True)]

model.fit(x=x_train, y=y_train, batch_size= 32, epochs=100, validation_data=(x_test, y_test), callbacks=callback)