import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import datasets, layers, models, utils

tfd = tfp.distributions

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = 2 * (train_images/255.0) - 1
test_images = 2 * (test_images/255.0) - 1
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
train_labels = utils.to_categorical(train_labels)
test_labels = utils.to_categorical(test_labels)

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def create_model_det():

    model = models.Sequential()
    model.add(layers.Conv2D(6, kernel_size=5, activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, kernel_size=5, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation='relu'))
    model.add(layers.Dense(84, activation='relu'))
    model.add(layers.Dense(10, activation=tf.nn.softmax))

    optimizer = tf.keras.optimizers.Adam(lr=0.001)

    model.compile(optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model



def create_model_bay():
    kl_divergence_fn = (lambda q, p, _: tfd.kl_divergence(q, p) / tf.cast(60000, dtype=tf.float32))

    model = models.Sequential()

    model.add(tfp.layers.Convolution2DFlipout(
        6, kernel_size=5, padding='SAME',
        kernel_divergence_fn=kl_divergence_fn,
        activation='relu', input_shape=(28, 28, 1)))

    model.add(tf.keras.layers.MaxPooling2D(
        pool_size=[2, 2], strides=[2, 2],
        padding='SAME'))

    model.add(tfp.layers.Convolution2DFlipout(
        16, kernel_size=5, padding='SAME',
        kernel_divergence_fn=kl_divergence_fn,
        activation='relu'))

    model.add(tf.keras.layers.MaxPooling2D(
        pool_size=[2, 2], strides=[2, 2],
        padding='SAME'))

    model.add(tf.keras.layers.Flatten())

    model.add(tfp.layers.DenseFlipout(
        120, kernel_divergence_fn=kl_divergence_fn,
        activation='relu'))

    model.add(tfp.layers.DenseFlipout(
        84, kernel_divergence_fn=kl_divergence_fn,
        activation='relu'))

    model.add(tfp.layers.DenseFlipout(
        10, kernel_divergence_fn=kl_divergence_fn,
        activation=tf.nn.softmax))

    optimizer = tf.keras.optimizers.Adam(lr=0.001)

    model.compile(optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def main():

    model_det = create_model_det()
    #model_det.load_weights('./model_weights/mnist_det')
    history = model_det.fit(train_images, train_labels, epochs=10,
                            validation_data=(test_images, test_labels))
    model_det.save_weights('./model_weights/mnist_det')

    f = open('./logs/mnist_det.txt', 'a')
    f.write(str(history.history))
    f.write('\n')
    f.close()

    model_bay = create_model_bay()
    #model_bay.load_weights('./model_weights/mnist_bay')
    history = model_bay.fit(train_images, train_labels, epochs=10,
                  validation_data=(test_images, test_labels))
    model_bay.save_weights('./model_weights/mnist_bay')

    f = open('./logs/mnist_bay.txt', 'a')
    f.write(str(history.history))
    f.write('\n')
    f.close()



if __name__ == "__main__": main()