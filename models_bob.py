import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers, models, utils, Model

tfd = tfp.distributions


def get_data_bob():
    train_dataset = utils.image_dataset_from_directory('./bob_dataset/train', label_mode='categorical',
                                                       image_size=(32, 32))
    test_dataset = utils.image_dataset_from_directory('./bob_dataset/test', label_mode='categorical',
                                                      image_size=(32, 32))

    normalization_layer = layers.Rescaling(1./255)
    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    return train_dataset, test_dataset


def load_cifar_model_det():

    cifar_model = models.Sequential()
    cifar_model.add(layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(32, 32, 3)))
    cifar_model.add(layers.Conv2D(32, kernel_size=3, activation='relu'))
    cifar_model.add(layers.MaxPooling2D((2, 2)))
    cifar_model.add(layers.Dropout(0.2))
    cifar_model.add(layers.Conv2D(64, kernel_size=3, activation='relu'))
    cifar_model.add(layers.Conv2D(64, kernel_size=3, activation='relu'))
    cifar_model.add(layers.MaxPooling2D((2, 2)))
    cifar_model.add(layers.Dropout(0.2))
    cifar_model.add(layers.Conv2D(128, kernel_size=3, activation='relu'))
    cifar_model.add(layers.Conv2D(128, kernel_size=3, activation='relu'))
    cifar_model.add(layers.Flatten())
    cifar_model.add(layers.Dropout(0.2))
    cifar_model.add(layers.Dense(512, activation='relu'))
    cifar_model.add(layers.Dropout(0.3))
    cifar_model.add(layers.Dense(10, activation=tf.nn.softmax))

    optimizer = tf.keras.optimizers.Adam(lr=0.001)

    cifar_model.compile(optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    cifar_model.load_weights('./model_weights/cifar_det_hist').expect_partial()

    x = layers.Dense(2, activation=tf.nn.softmax)(cifar_model.layers[-2].output)
    bob_model = Model(inputs=cifar_model.input, outputs=x)

    bob_model.compile(optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    return bob_model


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
    model.add(layers.Dense(2, activation=tf.nn.softmax))
    #model.add(layers.Dense(1, activation="sigmoid"))

    optimizer = tf.keras.optimizers.Adam(lr=0.001)

    model.compile(optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def load_cifar_model_bay():

    kl_divergence_fn = (lambda q, p, _: tfd.kl_divergence(q, p) / tf.cast(850, dtype=tf.float32))

    cifar_model = models.Sequential()
    cifar_model.add(tfp.layers.Convolution2DFlipout(64, kernel_size=11, kernel_divergence_fn=kl_divergence_fn,
                                              activation='relu', padding='same', strides=4, input_shape=(32, 32, 3)))
    cifar_model.add(layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='same'))
    cifar_model.add(tfp.layers.Convolution2DFlipout(64, kernel_size=5, kernel_divergence_fn=kl_divergence_fn,
                                              activation='relu', padding='same'))
    cifar_model.add(layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='same'))
    cifar_model.add(tfp.layers.Convolution2DFlipout(192, kernel_size=3, kernel_divergence_fn=kl_divergence_fn,
                                              activation='relu', padding='same'))
    cifar_model.add(tfp.layers.Convolution2DFlipout(384, kernel_size=3, kernel_divergence_fn=kl_divergence_fn,
                                              activation='relu', padding='same'))
    cifar_model.add(tfp.layers.Convolution2DFlipout(256, kernel_size=3, kernel_divergence_fn=kl_divergence_fn,
                                              activation='relu', padding='same'))
    cifar_model.add(layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='same'))
    cifar_model.add(layers.Flatten())
    cifar_model.add(tfp.layers.DenseFlipout(128, kernel_divergence_fn=kl_divergence_fn, activation='relu'))
    cifar_model.add(tfp.layers.DenseFlipout(10, kernel_divergence_fn=kl_divergence_fn, activation=tf.nn.softmax))

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=False)

    cifar_model.compile(optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'], experimental_run_tf_function=False)

    cifar_model.load_weights('./model_weights/cifar_bay_hist').expect_partial()

    kl_divergence_fn = (lambda q, p, _: tfd.kl_divergence(q, p) / tf.cast(850, dtype=tf.float32))

    x = tfp.layers.DenseFlipout(2, kernel_divergence_fn=kl_divergence_fn, activation=tf.nn.softmax)(cifar_model.layers[-2].output)
    bob_model = Model(inputs=cifar_model.input, outputs=x)

    bob_model.compile(optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    return bob_model


def create_model_bay():

    kl_divergence_fn = (lambda q, p, _: tfd.kl_divergence(q, p) / tf.cast(850, dtype=tf.float32))

    model = models.Sequential()
    model.add(tfp.layers.Convolution2DFlipout(64, kernel_size=11, kernel_divergence_fn=kl_divergence_fn,
                                              activation='relu', padding='same', strides=4, input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='same'))
    model.add(tfp.layers.Convolution2DFlipout(64, kernel_size=5, kernel_divergence_fn=kl_divergence_fn,
                                              activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='same'))
    model.add(tfp.layers.Convolution2DFlipout(192, kernel_size=3, kernel_divergence_fn=kl_divergence_fn,
                                              activation='relu', padding='same'))
    model.add(tfp.layers.Convolution2DFlipout(384, kernel_size=3, kernel_divergence_fn=kl_divergence_fn,
                                              activation='relu', padding='same'))
    model.add(tfp.layers.Convolution2DFlipout(256, kernel_size=3, kernel_divergence_fn=kl_divergence_fn,
                                              activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='same'))
    model.add(layers.Flatten())
    model.add(tfp.layers.DenseFlipout(128, kernel_divergence_fn=kl_divergence_fn, activation='relu'))
    model.add(tfp.layers.DenseFlipout(2, kernel_divergence_fn=kl_divergence_fn, activation=tf.nn.softmax))
    #model.add(tfp.layers.DenseFlipout(1, kernel_divergence_fn=kl_divergence_fn, activation='sigmoid'))

    #optimizer = tf.keras.optimizers.Adam(lr=0.001)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=False)

    model.compile(optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'], experimental_run_tf_function=False)

    return model


def main():

    train_dataset, test_dataset = get_data_bob()

    model_det = load_cifar_model_det()
    #model_det = create_model_det()
    #model_det.load_weights('./model_weights/bob_det')
    history = model_det.fit(train_dataset, epochs=10, validation_data=test_dataset)
    model_det.save_weights('./model_weights/bob_det')

    f = open('./logs/bob_det.txt', 'a')
    f.write(str(history.history))
    f.write('\n')
    f.close()

    model_bay = load_cifar_model_bay()
    #model_bay = create_model_bay()
    #model_bay.load_weights('./model_weights/bob_bay')
    history = model_bay.fit(train_dataset, epochs=10, validation_data=test_dataset)
    model_bay.save_weights('./model_weights/bob_bay')

    f = open('./logs/bob_bay.txt', 'a')
    f.write(str(history.history))
    f.write('\n')
    f.close()

    print("\ndone")


if __name__ == "__main__": main()
