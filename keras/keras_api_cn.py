import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math

from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from .mnist  import MNIST


data = MNIST(data_dir='data/MNIST/')
print("Size of:")
print("- Training-set:\t\t{}".format(data.num_train))
print("- Validation-set:\t{}".format(data.num_val))
print("- Test-set:\t\t{}".format(data.num_test))

# The number of pixels in each dimension of an image.
img_size = data.img_size

# The images are stored in one-dimensional arrays of this length.
img_size_flat = data.img_size_flat

# Tuple with height and width of images used to reshape arrays.
img_shape = data.img_shape

# Tuple with height, width and depth used to reshape arrays.
# This is used for reshaping in Keras.
img_shape_full = data.img_shape_full

# Number of classes, one class for each of 10 digits.
num_classes = data.num_classes

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = data.num_channels

def plot_images(images, cls_true, cls_pred=None):
    assert  len(images) == len(cls_true)

    fig, axes = plt.subplot(3, 3)
    fig.suplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        if cls_pred is None:
            xlabel = 'True: {0}'.format(cls_true[i])
        else:
            xlabel = 'True: {o}, pred: {1}'.format(cls_true[i], cls_pred[i])
        ax.set_xlabel(xlabel)

        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

images = data.x_test[0:9]
cls_true = data.y_test_cls[0:9]
plot_images(images=images, cls_true=cls_true)

def plot_example_errors(cls_pred):
    incorrect = (cls_pred != data.y_test_cls)
    images = data.x_test[incorrect]
    cls_pred = cls_pred[incorrect]

    cls_true = data.y_test_cls[incorrect]

    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])

# 顺序模型（Sequential Model）
model = Sequential()

model.add(InputLayer(input_shape=(img_size_flat,)))

model.add(Reshape(img_shape_full))

model.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='same', activation='relu', name='layer_conv1'))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Conv2D(kernel_size=5, strides=1, filters=36, padding='same', activation='relu', name='layer_conv2'))
model.add(MaxPooling2D(pool_size=2, strides=2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

optimizer = Adam(lr=0.03)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accracy'])

model.fit(x=data.x_train,
          y=data.y_train)
result = model.evaluate(x=data.x_test, y=data.y_test)
for name, value in zip(model.metrics_names[1], result):
    print(name, value)

print("{0}: {1:.2%}".format(model.metrics_names[1], result[1]))

image = data.x_test[0:9]
cls_true = data.y_test_cls[0:9]
y_pred = model.predict(x=image)
cls_pred = np.argmax(y_pred, axis=1)
plot_images(images=image, cls_true=cls_true, cls_pred=cls_pred)

y_pred = model.predict(x=data.x_test)
cls_pred = np.argmax(y_pred, axis=1)
plot_example_errors(cls_pred)

# 函数试模型
inputs = Input(shape=(img_size_flat,))
net = inputs

net = Reshape(img_shape_full)(net)

net = Conv2D(kernel_size=5, strides=1, filters=16, padding='same',activation='relu', name='layer_conv1')(net)
net = MaxPooling2D(strides=2, pool_size=2)(net)

net = Conv2D(kernel_size=5, strides=1, filters=64, padding='same',activation='relu', name='layer_conv2')(net)
net = MaxPooling2D(strides=2, pool_size=2)

net = Flatten()(net)
net = Dense(128, activation='relu')(net)
net = Dense(num_classes, activation='softmax')(net)
outputs = net

# 模型编译
# 创建一个新的Keras函数式模型的实例
model2 = Model(inputs=inputs, outputs=outputs)
model2.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model2.fit(x=data.x_train, y=data.y_train, epochs=1, batch_size=128)

result = model2.evaluate(x=data.x_test, y=data.y_test)
for name, value in zip(model2.metrics_names[1], result):
    print(name, value)

print("{0}: {1:.2%}".format(model2.metrics_names[1], result[1]))

y_pred = model2.predict(x=data.x_test)
cls_pred = np.argmax(y_pred, axis=1)
plot_example_errors(cls_pred)

path_model = 'model.keras'
model2.save(path_model)
del model2
model3 = load_model(path_model)