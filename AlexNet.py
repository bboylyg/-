# AlexNet
from keras import layers
from keras.models import Model
from keras.utils import plot_model

def alexnet(in_shape=(227,227,3), n_classes=1000, opt='sgd'):
    in_layer = layers.Input(in_shape)
    conv1 = layers.Conv2D(filters=96, kernel_size=11, strides=4, activation='relu')(in_layer)
    pool1 = layers.MaxPool2D(pool_size=3, strides=2)(conv1)
    conv2 = layers.Conv2D(filters=256, kernel_size=5, strides=1, padding='same', activation='relu')(pool1)
    pool2 = layers.MaxPool2D(pool_size=3, strides=2)(conv2)
    conv3 = layers.Conv2D(filters=384, kernel_size=3, strides=1, padding='same', activation='relu')(pool2)
    conv4 = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')(conv3)
    pool3 = layers.MaxPool2D(pool_size=3, strides=2)(conv4)
    flattened = layers.Flatten()(pool3)
    dense1 = layers.Dense(4096, activation='relu')(flattened)
    drop1 = layers.Dropout(0.5)(dense1)
    dense2 = layers.Dense(4096, activation='relu')(drop1)
    drop2 = layers.Dropout(0.5)(dense2)
    preds = layers.Dense(n_classes, activation='softmax')(drop2)
    
    model = Model(in_layer, preds)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                 metrics=["accuracy"])
    return model

if __name__ == '__main__':
    model = alexnet()
    #plot_model(model, to_file='F://alexnet_plot.png', show_shapes=True, show_layer_names=True)
    print(model.summary())
