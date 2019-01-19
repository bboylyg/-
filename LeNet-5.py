# LeNet-5, a 7-layer convolutional neural network
from keras import layers
from keras.models import Model
from keras.utils import plot_model

def lenet_5(in_shape=(32, 32, 1), n_classes=10, opt='sgd'):
    in_layer = layers.Input(in_shape)
    conv1 = layers.Conv2D(filters=6, kernel_size=5, padding='same', activation='relu')(in_layer)
    pool1 = layers.MaxPool2D()(conv1)
    conv2 = layers.Conv2D(filters=6, kernel_size=5, padding='same', activation='relu')(pool1)
    pool2 = layers.MaxPool2D()(conv2)
    flatten = layers.Flatten()(pool2)
    dense1 = layers.Dense(84, activation='relu')(flatten)
    preds = layers.Dense(n_classes, activation='softmax')(dense1)

    model = Model(in_layer, preds)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model

if __name__ == '__main__':
    model = lenet_5()
    #plot_model(model, to_file='F://lenet5_plot.png', show_shapes=True, show_layer_names=True)
    print(model.summary())
