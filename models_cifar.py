import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import datasets, layers, models, utils

tfd = tfp.distributions


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0
train_labels = utils.to_categorical(train_labels)
test_labels = utils.to_categorical(test_labels)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


def create_model_det():

    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.Conv2D(32, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, kernel_size=3, activation='relu'))
    model.add(layers.Conv2D(64, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(128, kernel_size=3, activation='relu'))
    model.add(layers.Conv2D(128, kernel_size=3, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(10, activation=tf.nn.softmax))

    optimizer = tf.keras.optimizers.Adam(lr=0.001)

    model.compile(optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def create_model_bay():

    kl_divergence_fn = (lambda q, p, _: tfd.kl_divergence(q, p) / tf.cast(len(train_images), dtype=tf.float32))

    model = models.Sequential()
    model.add(tfp.layers.Convolution2DFlipout(64, kernel_size=11, kernel_divergence_fn=kl_divergence_fn, activation='relu', padding='same', strides=4, input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='same'))
    model.add(tfp.layers.Convolution2DFlipout(64, kernel_size=5, kernel_divergence_fn=kl_divergence_fn, activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='same'))
    model.add(tfp.layers.Convolution2DFlipout(192, kernel_size=3, kernel_divergence_fn=kl_divergence_fn, activation='relu', padding='same'))
    model.add(tfp.layers.Convolution2DFlipout(384, kernel_size=3, kernel_divergence_fn=kl_divergence_fn, activation='relu', padding='same'))
    model.add(tfp.layers.Convolution2DFlipout(256, kernel_size=3, kernel_divergence_fn=kl_divergence_fn, activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='same'))
    model.add(layers.Flatten())
    model.add(tfp.layers.DenseFlipout(128, kernel_divergence_fn=kl_divergence_fn, activation='relu'))
    model.add(tfp.layers.DenseFlipout(10, kernel_divergence_fn=kl_divergence_fn, activation=tf.nn.softmax))

    #optimizer = tf.keras.optimizers.Adam(lr=0.001)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=False)

    model.compile(optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'], experimental_run_tf_function=False)

    return model


def main():
    model_det = create_model_det()
    #model_det.load_weights('./model_weights/cifar_det')
    history = model_det.fit(train_images, train_labels, epochs=10,
                  validation_data=(test_images, test_labels))
    model_det.save_weights('./model_weights/cifar_det')

    f = open('./logs/cifar_bay.txt', 'a')
    f.write(str(history.history))
    f.write('\n')
    f.close()

    model_bay = create_model_bay()
    #model_bay.load_weights('./model_weights/cifar_bay')
    history = model_bay.fit(train_images, train_labels, epochs=10,
                  validation_data=(test_images, test_labels))
    model_bay.save_weights('./model_weights/cifar_bay')

    f = open('./logs/cifar_det.txt', 'a')
    f.write(str(history.history))
    f.write('\n')
    f.close()



if __name__ == "__main__": main()
