from tensorflow.python.keras.models import Model, Input
from tensorflow.python.keras.layers import Dropout, Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import AvgPool2D, concatenate

class InceptionV3(object):
    def __init__(self, input_shape=None, weight_decay=0, nb_classes=0):
        self.input_shape = input_shape
        self.weight_decay = weight_decay
        self.classes = nb_classes

    def block1_module1(self, net, name):
        # 1x1
        net_1x1 = Conv2D(filters=64, kernel_size=1, padding='same', activation='relu',
                         name=name + '_1x1',
                         kernel_regularizer=regularizers.l2(self.weight_decay))(net)
        net_1x1 = BatchNormalization()(net_1x1)

        # 5x5
        net_5x5 = Conv2D(filters=48, kernel_size=1, padding='same', activation='relu',
                         name=name + '_5x5',
                         kernel_regularizer=regularizers.l2(self.weight_decay))(net)
        net_5x5 = BatchNormalization(net_5x5)
        net_5x5 = Conv2D(filters=64, kernel_size=5, padding='same', activation='relu',
                         name=name + '_5x5_2',
                         kernel_regularizer=regularizers.l2(self.weight_decay))(net_5x5)
        net_5x5 = BatchNormalization()(net_5x5)

        # 3x3
        net_3x3 = Conv2D(filters=64, kernel_size=1,padding='same', activation='relu',
                         name=name + '_3x3',
                         kernel_regularizer=regularizers.l2(self.weight_decay))(net)
        net_3x3 = BatchNormalization()(net_3x3)
        net_3x3 = Conv2D(filters=96, kernel_size=3, activation='relu', padding='same',
                         name=name + '_3x3_2',
                         kernel_regularizer=regularizers.l2(self.weight_decay))(net_3x3)
        net_3x3 = BatchNormalization()(net_3x3)
        net_3x3 = Conv2D(filters=96, kernel_size=3, activation='relu', padding='same',
                         name=name + '_3x3_3',
                         kernel_regularizer=regularizers.l2(self.weight_decay))(net_3x3)
        net_3x3 = BatchNormalization()(net_3x3)

        # 1x1xavg
        net_1x1_avg = AvgPool2D(pool_size=3, strides=1, padding='same',
                                name=name + '_net_1x1_avg')(net)
        net_1x1_avg = Conv2D(filters=32, kernel_size=1, activation='relu', padding='same',
                             name=name + '_net_1x1_avg_conv1',
                             kernel_regularizer=regularizers.l2(self.weight_decay))(net_1x1_avg)
        net = concatenate([net_1x1, net_5x5, net_3x3, net_1x1_avg], axis=-1, name=name + '_mixed')
        return  net

    def block1_module2(self, net, name):
        # 1x1
        net_1x1 = Conv2D(filters=64, kernel_size=1, padding='same', activation='relu',
                         name=name + '_1x1',
                         kernel_regularizer=regularizers.l2(self.weight_decay))(net)
        net_1x1 = BatchNormalization()(net_1x1)

        # 5x5
        net_5x5 = Conv2D(filters=48, kernel_size=1, padding='same', activation='relu',
                         name=name + '_5x5',
                         kernel_regularizer=regularizers.l2(self.weight_decay))(net)
        net_5x5 = BatchNormalization(net_5x5)
        net_5x5 = Conv2D(filters=64, kernel_size=5, padding='same', activation='relu',
                         name=name + '_5x5_2',
                         kernel_regularizer=regularizers.l2(self.weight_decay))(net_5x5)
        net_5x5 = BatchNormalization()(net_5x5)

        # 3x3
        net_3x3 = Conv2D(filters=64, kernel_size=1,padding='same', activation='relu',
                         name=name + '_3x3',
                         kernel_regularizer=regularizers.l2(self.weight_decay))(net)
        net_3x3 = BatchNormalization()(net_3x3)
        net_3x3 = Conv2D(filters=96, kernel_size=3, activation='relu', padding='same',
                         name=name + '_3x3_2',
                         kernel_regularizer=regularizers.l2(self.weight_decay))(net_3x3)
        net_3x3 = BatchNormalization()(net_3x3)
        net_3x3 = Conv2D(filters=96, kernel_size=3, activation='relu', padding='same',
                         name=name + '_3x3_3',
                         kernel_regularizer=regularizers.l2(self.weight_decay))(net_3x3)
        net_3x3 = BatchNormalization()(net_3x3)

        # 1x1xavg
        net_1x1_avg = AvgPool2D(pool_size=3, strides=1, padding='same',
                                name=name + '_net_1x1_avg')(net)
        net_1x1_avg = Conv2D(filters=64, kernel_size=1, activation='relu', padding='same',
                             name=name + '_net_1x1_avg_conv1',
                             kernel_regularizer=regularizers.l2(self.weight_decay))(net_1x1_avg)
        net = concatenate([net_1x1, net_5x5, net_3x3, net_1x1_avg], axis=-1, name=name + '_mixed')
        return  net

    def block2_module1(self, net):
        # 3x3
        net_3x3 = Conv2D(filters=384, strides=2, kernel_size=3, padding='same',
                     name='bkock2_module1_conv1',activation='relu',
                     kernel_regularizer=regularizers.l2(self.weight_decay))(net)
        net_3x3 = BatchNormalization()(net_3x3)

        # 1x1_3x3
        net_1x1_3x3 = Conv2D(filters=64, kernel_size=1, padding='same', activation='relu',
                             name='block2_module1_net_1x1_conv1',
                             kernel_regularizer=regularizers.l2(self.weight_decay))(net)
        net_1x1_3x3 = BatchNormalization()(net_1x1_3x3)
        net_1x1_3x3 = Conv2D(filters=96, kernel_size=3, activation='relu', padding='same',
                             name='block2_module1_net_1x1_conv2',
                             kernel_regularizer=regularizers.l2(self.weight_decay))(net_1x1_3x3)
        net_1x1_3x3 = BatchNormalization()(net_1x1_3x3)
        net_1x1_3x3 = Conv2D(filters=96, kernel_size=3, activation='relu', padding='same', strides=2,
                             name='block2_module1_net_1x1_conv3',
                             kernel_regularizer=regularizers.l2(self.weight_decay))(net_1x1_3x3)

        net_max = MaxPooling2D(pool_size=3, strides=2, padding='same', name='block2_module1_max')(net)
        net = concatenate([net_3x3, net_1x1_3x3, net_max], axis=-1, name='block2_module1_mixed')
        return net

    def block2_modul3_4(self, net, name):
        # 1x1
        net_1x1 = Conv2D(filters=192, kernel_size=1, padding='same', activation='relu',
                         name=name + "_1x1_conv1",
                         kernel_regularizer=regularizers.l2(self.weight_decay))(net)
        net_1x1 = BatchNormalization()(net_1x1)

        # 1x7
        net_1x7 = Conv2D(filters=160, kernel_size=(1, 1), padding='same', activation='relu',
                         name=name+'_1x7_conv1',
                         kernel_regularizer=regularizers.l2(self.weight_decay))(net)
        net_1x7 = BatchNormalization()(net_1x7)
        net_1x7 = Conv2D(filters=160, kernel_size=(1, 7), padding='same', activation='relu',
                         name=name + '_1x7_conv2',
                         kernel_regularizer=regularizers.l2(self.weight_decay))(net_1x7)
        net_1x7 = BatchNormalization()(net_1x7)
        net_1x7 = Conv2D(filters=192, kernel_size=(7, 1), padding='same', activation='relu',
                         name=name+'_1x7_conv3',
                         kernel_regularizer=regularizers.l2(self.weight_decay))(net_1x7)
        net_1x7 = BatchNormalization()(net_1x7)

        net_7x1 = Conv2D(filters=160, kernel_size=(1, 1), padding='same', activation='relu',
                         name=name+'_7x1_conv1',
                         kernel_regularizer=regularizers.l2(self.weight_decay))(net)
        net_7x1 = Conv2D(filters=160, kernel_size=(7, 1), padding='same', activation='relu',
                         name=name+'_7x1_conv2',
                         kernel_regularizer=regularizers.l2(self.weight_decay))(net_7x1)
        net_7x1 = Conv2D(filters=160, kernel_size=(7, 1), padding='same', activation='relu',
                         name=name+'_7x1_conv3',
                         kernel_regularizer=regularizers.l2(self.weight_decay))(net_7x1)
        net_7x1 = Conv2D(filters=192, kernel_size=(1, 7), padding='same', activation='relu',
                         name=name+'_7x1_conv4',
                         kernel_regularizer=regularizers.l2(self.weight_decay))(net_7x1)

        net_avg = AvgPool2D(pool_size=3, strides=1, padding='same', name=name+'_1x1_avg')(net)
        net_avg = Conv2D(filters=192, kernel_size=1, padding='same', activation='relu',
                         name=name+'_1x1_avg_conv1',
                         kernel_regularizer=regularizers.l2(self.weight_decay))(net_avg)
        net = concatenate([net_1x1, net_1x7, net_7x1, net_avg], axis=-1)

        return net

    def bulid_model(self):
        inputs = Input(shape=self.input_shape)
        net = inputs
        # block1
        net = Conv2D(filters=32, kernel_size=3, strides=2, activation='relu', padding='same',
                     name='bock1_conv1', kernel_regularizer=regularizers.l2(self.weight_decay))(net)
        net = BatchNormalization()(net)
        net = Conv2D(filters=32, kernel_size=3, activation='relu', padding='same',
                     name='block1_conv2', kernel_regularizer=regularizers.l2(self.weight_decay)(net))
        net = BatchNormalization()(net)
        net = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same',
                     name='block1_conv3', kernel_regularizer=regularizers.l2(self.weight_decay))(net)
        net = BatchNormalization()(net)
        net = MaxPooling2D(pool_size=3, strides=2, padding='same', name='block1_pool')(net)

        #block2
        net = Conv2D(filters=80, kernel_size=1,activation='relu',padding='same',
                     name='block2_conv1', kernel_regularizer=regularizers.l2(self.weight_decay))(net)
        net = BatchNormalization()(net)
        net = Conv2D(filters=192, kernel_size=3, activation='relu', padding='same',
                     name='block2_conv2', kernel_regularizer=regularizers.l2(self.weight_decay))(net)
        net = BatchNormalization()(net)
        net = MaxPooling2D(pool_size=3, strides=2, padding='same', name='block2_pool')(net)

        net = self.block1_module1(net, 'block1_module1')
        net = self.block1_module2(net, "block1_module2")
        net = self.block1_module2(net, 'block1_module2_1')
        net = self.block2_module1(net)

        # 1x1
        net_1x1 = Conv2D(filters=128, kernel_size=1, padding='same', activation='relu',
                         name='block2_module2_1x1',
                         kernel_regularizer=regularizers.l2(self.weight_decay))(net)
        net_1x1 = BatchNormalization()(net_1x1)

        # 1x7
        net_1x7 = Conv2D(filters=128, kernel_size=(1,1), padding='same', activation='relu',
                         name='block2_module2_1x7_conv1',
                         kernel_regularizer=regularizers.l2(self.weight_decay))(net)
        net_1x7 = BatchNormalization()(net_1x7)
        net_1x7 = Conv2D(filters=128, kernel_size=(1,7), padding='same', activation='relu',
                         name='block2_module2_1x7_conv2',
                         kernel_regularizer=regularizers.l2(self.weight_decay))(net_1x7)
        net_1x7 = BatchNormalization()(net_1x7)
        net_1x7 = Conv2D(filters=192, kernel_size=(7,1), padding='same', activation='relu',
                         name='block2_module2_1x7_conv3',
                         kernel_regularizer=regularizers.l2(self.weight_decay))(net_1x7)
        net_1x7 = BatchNormalization()(net_1x7)

        net_7x1 = Conv2D(filters=128, kernel_size=(1,1), padding='same', activation='relu',
                         name='block2_module2_7x1_conv1',
                         kernel_regularizer=regularizers.l2(self.weight_decay))(net)
        net_7x1 = Conv2D(filters=128, kernel_size=(7, 1), padding='same', activation='relu',
                         name='block2_module2_7x1_conv2',
                         kernel_regularizer=regularizers.l2(self.weight_decay))(net_7x1)
        net_7x1 = Conv2D(filters=128, kernel_size=(7, 1), padding='same', activation='relu',
                         name='block2_module2_7x1_conv3',
                         kernel_regularizer=regularizers.l2(self.weight_decay))(net_7x1)
        net_7x1 = Conv2D(filters=192, kernel_size=(1, 7), padding='same', activation='relu',
                         name='block2_module2_7x1_conv4',
                         kernel_regularizer=regularizers.l2(self.weight_decay))(net_7x1)

        net_avg = AvgPool2D(pool_size=3, strides=1, padding='same', name='block2_module2_1x1_avg')(net)
        net_avg = Conv2D(filters=192, kernel_size=1, padding='same', activation='relu',
                         name='block2_module2_1x1_avg_conv1',
                         kernel_regularizer=regularizers.l2(self.weight_decay))(net_avg)
        net = concatenate([net_1x1, net_1x7, net_7x1, net_avg], axis=-1)

        net = self.block2_modul3_4(net, 'block2_module3')
        net = self.block2_modul3_4(net, 'block2_module4')

        # 1x1
        net_1x1 = Conv2D(filters=192, kernel_size=1, padding='same', activation='relu',
                         name='block2_module5_1x1',
                         kernel_regularizer=regularizers.l2(self.weight_decay))(net)
        net_1x1 = BatchNormalization()(net_1x1)

        # 1x7
        net_1x7 = Conv2D(filters=192, kernel_size=(1, 1), padding='same', activation='relu',
                         name='block2_module5_1x7_conv1',
                         kernel_regularizer=regularizers.l2(self.weight_decay))(net)
        net_1x7 = BatchNormalization()(net_1x7)
        net_1x7 = Conv2D(filters=192, kernel_size=(1, 7), padding='same', activation='relu',
                         name='block2_module5_1x7_conv2',
                         kernel_regularizer=regularizers.l2(self.weight_decay))(net_1x7)
        net_1x7 = BatchNormalization()(net_1x7)
        net_1x7 = Conv2D(filters=192, kernel_size=(7, 1), padding='same', activation='relu',
                         name='block2_module5_1x7_conv3',
                         kernel_regularizer=regularizers.l2(self.weight_decay))(net_1x7)
        net_1x7 = BatchNormalization()(net_1x7)

        net_7x1 = Conv2D(filters=192, kernel_size=(1, 1), padding='same', activation='relu',
                         name='block2_module5_7x1_conv1',
                         kernel_regularizer=regularizers.l2(self.weight_decay))(net)
        net_7x1 = Conv2D(filters=192, kernel_size=(7, 1), padding='same', activation='relu',
                         name='block2_module5_7x1_conv2',
                         kernel_regularizer=regularizers.l2(self.weight_decay))(net_7x1)
        net_7x1 = Conv2D(filters=192, kernel_size=(7, 1), padding='same', activation='relu',
                         name='block2_module5_7x1_conv3',
                         kernel_regularizer=regularizers.l2(self.weight_decay))(net_7x1)
        net_7x1 = Conv2D(filters=192, kernel_size=(1, 7), padding='same', activation='relu',
                         name='block2_module5_7x1_conv4',
                         kernel_regularizer=regularizers.l2(self.weight_decay))(net_7x1)

        net_avg = AvgPool2D(pool_size=3, strides=1, padding='same', name='block2_module5_1x1_avg')(net)
        net_avg = Conv2D(filters=192, kernel_size=1, padding='same', activation='relu',
                         name='block2_module5_1x1_avg_conv1',
                         kernel_regularizer=regularizers.l2(self.weight_decay))(net_avg)
        net = concatenate([net_1x1, net_1x7, net_7x1, net_avg], axis=-1)










