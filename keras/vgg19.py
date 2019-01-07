from tensorflow.python.keras.models import Sequential, InputLayer
from tensorflow.python.keras.layers import Dropout, Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras import regularizers


class VGG19(object):
    def __init__(self, input_shape=None, weight_decay=0, nb_classes=None):
        self.input_shape = input_shape
        self.weigth_decay = weight_decay
        self.classes = nb_classes

    def bulid_model(self, batchnormaliztion = False):
        model = Sequential()
        model.add(InputLayer(input_shape=self.input_shape))
        # block1
        model.add(Conv2D(filters=64, kernel_size=3, padding='same',
                         name='block1_conv1', activation='relu',
                         kernel_regularizer=regularizers.l2(self.weigth_decay)))
        model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same',
                         name='bock1_conv2', kernel_regularizer=regularizers.l2(self.weigth_decay)))
        model.add(MaxPooling2D(pool_size=2, strides=2, padding='same', name='block1_pool'))

        if not batchnormaliztion:
            model.add(BatchNormalization(name='block1_batchnormalize'))

        #block2
        model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu',
                         name='block2_conv1', kernel_regularizer=regularizers.l2(self.weigth_decay)))
        model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu',
                         name='block2_conv2', kernel_regularizer=regularizers.l2(self.weigth_decay)))
        model.add(MaxPooling2D(pool_size=2, strides=2, padding='same', name='block2_pool'))

        if not batchnormaliztion:
            model.add(BatchNormalization(name='block2_batchnormalize'))

        #block3
        model.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',
                         name='block3_conv1', kernel_regularizer=regularizers.l2(self.weigth_decay)))
        model.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',
                         name='block3_conv2', kernel_regularizer=regularizers.l2(self.weigth_decay)))
        model.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',
                         name='block3_conv3', kernel_regularizer=regularizers.l2(self.weigth_decay)))
        model.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',
                         name='block3_conv4', kernel_regularizer=regularizers.l2(self.weigth_decay)))
        model.add(MaxPooling2D(pool_size=2, strides=2, padding='same', name='block3_pool'))

        if not batchnormaliztion:
            model.add(BatchNormalization(name='block3_batchnormalize'))

        # block4
        model.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='relu',
                         name='block4_conv1', kernel_regularizer=regularizers.l2(self.weigth_decay)))
        model.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='relu',
                         name='block4_conv2', kernel_regularizer=regularizers.l2(self.weigth_decay)))
        model.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='relu',
                         name='block4_conv3', kernel_regularizer=regularizers.l2(self.weigth_decay)))
        model.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='relu',
                         name='block4_conv4', kernel_regularizer=regularizers.l2(self.weigth_decay)))
        model.add(MaxPooling2D(pool_size=2, strides=2, padding='same', name='block4_pool'))

        if not batchnormaliztion:
            model.add(BatchNormalization(name='block4_batchnormalize'))

        #block5
        model.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='relu',
                         name='block5_conv1', kernel_regularizer=regularizers.l2(self.weigth_decay)))
        model.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='relu',
                         name='block5_conv2', kernel_regularizer=regularizers.l2(self.weigth_decay)))
        model.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='relu',
                         name='block5_conv3', kernel_regularizer=regularizers.l2(self.weigth_decay)))
        model.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='relu',
                         name='block5_conv4', kernel_regularizer=regularizers.l2(self.weigth_decay)))
        model.add(MaxPooling2D(pool_size=2, strides=2, padding='same', name='block5_pool'))

        if not batchnormaliztion:
            model.add(BatchNormalization(name='block5_batchnormalize'))

        # fcy layer
        model.add(Flatten(name='Flatten'))
        model.add(Dense(4096, activation='relu', name='fc1',
                        kernel_regularizer=regularizers.l2(self.weigth_decay)))
        if not batchnormaliztion:
            model.add(BatchNormalization(name='fc1_batchnormalize'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu', name='fc2',
                        kernel_regularizer=regularizers.l2(self.weigth_decay)))
        if not batchnormaliztion:
            model.add(BatchNormalization(name='block2_batchnormalize'))
        model.add(Dropout(0.5))
        model.add(Dense(self.classes, activation='softmax', name='prediction'))
        self.model = model
        self.model.summary()


if __name__ == '__main__':
    model = VGG19(input_shape=(224, 224, 3), nb_classes=3)
    model.bulid_model(batchnormaliztion=True)