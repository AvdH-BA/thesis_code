from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
#import tensorflow.compat.v2 as tf
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import datasets, layers, models, utils

import statistics
import matplotlib.pyplot as plt

from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.tf2.attacks.momentum_iterative_method import momentum_iterative_method
from cleverhans.tf2.attacks.spsa import spsa

#tf.enable_v2_behavior()

tfd = tfp.distributions
tfpl = tfp.layers

x = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

class MNISTSequence(tf.keras.utils.Sequence):

  def __init__(self, data=None, batch_size=128):
    if data:
      images, labels = data
    else:
      images, labels = MNISTSequence.__generate_fake_data(
          num_images=128, num_classes=10)
    self.images, self.labels = MNISTSequence.__preprocessing(images, labels)
    self.batch_size = batch_size

  @staticmethod
  def __preprocessing(images, labels):
    images = 2 * (images / 255.) - 1.
    images = images[..., tf.newaxis]

    labels = tf.keras.utils.to_categorical(labels)
    return images, labels

  def __generate_fake_data(num_images, num_classes):
    images = np.random.randint(low=0, high=256,
                               size=(num_images, 28, 28))
    labels = np.random.randint(low=0, high=num_classes,
                               size=num_images)
    return images, labels

  def __len__(self):
    return int(tf.math.ceil(len(self.images) / self.batch_size))

  def __getitem__(self, idx):
    batch_x = self.images[idx * self.batch_size: (idx + 1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
    return batch_x, batch_y

def main():

    mnist = tf.keras.datasets.mnist

    batch_size = 128

    _, heldout_set = mnist.load_data()
    heldout_seq = MNISTSequence(data=heldout_set, batch_size=batch_size)


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

    model_bay = create_model_bay()
    model_det = create_model_det()

    model_det.load_weights('./model_weights/mnist_det').expect_partial()
    model_bay.load_weights('./model_weights/mnist_bay').expect_partial()


    def det_batch_spsa(batch_number=0, n_images=batch_size, thresholds=[0.0], figures=0, epsilon=0.1):
        test_images, test_labels = MNISTSequence.__getitem__(heldout_seq, batch_number)

        data_type = f"spsa{epsilon}"

        res = []

        x_32 = tf.cast(test_images, tf.float32)

        spsas = []

        for i in range(n_images):
            x_spsa = spsa(model_fn=model_det, x=x_32[i][None, :, :, :], y=test_labels[i], spsa_iters=1, delta=0.01,
                          eps=epsilon, nb_iter=100, clip_max=1.0, clip_min=-1.0, learning_rate=0.5, is_debug=False)
            spsas.append(x_spsa)


        for j in range(len(thresholds)):
            acc = 0
            pred_correct = 0.0
            pred_incorrect = 0.0
            guessed = 0

            for i in range(n_images):
                x_spsa = spsas[i]
                pred = model_det.predict(x_spsa, verbose=0)
                if np.max(pred[0]) >= thresholds[j]:
                    guessed += 1
                    if np.argmax(pred[0]) == np.argmax(test_labels[i]):
                        acc += 1
                        pred_correct += np.max(pred[0])
                    else:
                        pred_incorrect += np.max(pred[0])
            pred_correct = pred_correct / acc if acc > 0 else -1
            pred_incorrect = pred_incorrect / (guessed - acc) if guessed - acc > 0 else -1
            acc = acc / guessed if guessed > 0 else -1

            res.append((thresholds[j], acc, guessed / n_images, pred_correct, pred_incorrect))

            if j == 0:
                print(f"\n\n\nDeterministic model accuracy with {data_type} data:\n")

            if thresholds[j] == 0.0:
                print(f"Accuracy using no confidence threshold:\t\t\t\t{res[j][1] * 100:4.1f}%")
            else:
                print(f"Accuracy using a confidence threshold of {res[j][0]}:\t\t\t{res[j][1] * 100:4.1f}%"
                      f"\t({res[j][2] * 100:4.1f}% of images used)")
            print(f"Prediction confidence among correctly classified images:\t{res[j][3]:4.3f}")
            print(f"Prediction confidence among incorrectly classified images:\t{res[j][4]:4.3f}\n")

        for i in range(min(n_images, figures)):
            plt.imshow(tf.reshape(spsas[i], (28, 28)), cmap='gist_gray')
            plt.title(f"{data_type} image (det)")
            plt.savefig(f"./images_mnist/{data_type}_{i}_det_image.jpeg", bbox_inches='tight')
            plt.clf()
            plt.bar(x, pred[0])
            plt.title("prediction (det)")
            plt.ylim([0, 1])
            plt.savefig(f"./images_mnist/{data_type}_{i}_prediction.jpeg", bbox_inches='tight')
            plt.clf()

        return res

    def bay_batch_spsa_fb(batch_number=0, n_images=batch_size, thresholds=[0.0], figures=0, iterations=100, epsilon=0.1):
        data_type = f"spsa{epsilon}"

        test_images, test_labels = MNISTSequence.__getitem__(heldout_seq, batch_number)

        x_32 = tf.cast(test_images, tf.float32)

        model_outputs = np.zeros((iterations, n_images, 10))

        spsas = []

        for i in range(n_images):
            x_spsa = spsa(model_fn=model_bay, x=x_32[i][None, :, :, :], y=test_labels[i], spsa_iters=1, delta=0.01,
                          eps=epsilon, nb_iter=100, clip_max=1.0, clip_min=-1.0, learning_rate=0.5, is_debug=False)
            spsas.append(x_spsa)

        for i in range(iterations):
            for j in range(n_images):
                prediction = model_bay.predict(spsas[j], verbose=0)
                for k in range(10):
                    model_outputs[i][j][k] = prediction[0][k]

        means = np.mean(model_outputs, axis=0)
        mean_max = means.max(axis=-1)
        predictions = means.argmax(axis=-1)

        res = []

        for i in range(len(thresholds)):
            acc = 0
            mean_correct = 0.0
            mean_incorrect = 0.0
            guessed = 0

            for j in range(n_images):
                if mean_max[j] >= thresholds[i]:
                    guessed += 1
                    if predictions[j] == np.argmax(test_labels[j]):
                        acc += 1
                        mean_correct += mean_max[j]
                    else:
                        mean_incorrect += mean_max[j]

            mean_correct = mean_correct/acc if acc > 0 else -1
            mean_incorrect = mean_incorrect/(guessed-acc) if guessed - acc > 0 else -1
            acc = acc / guessed if guessed > 0 else -1

            res.append((thresholds[i], acc, guessed/n_images, mean_correct, mean_incorrect))

            if i == 0:
                print(f"\n\n\nBayesian model accuracy with {data_type} data:\n")

            if thresholds[i] == 0.0:
                print(f"Accuracy using no confidence threshold:\t\t\t\t{res[i][1] * 100:4.1f}%")
            else:
                print(f"Accuracy using a confidence threshold of {res[i][0]}:\t\t\t{res[i][1] * 100:4.1f}%"
                      f"\t({res[i][2] * 100:4.1f}% of images used)")
            print(f"Prediction confidence among correctly classified images:\t{res[i][3]:4.3f}")
            print(f"Prediction confidence among incorrectly classified images:\t{res[i][4]:4.3f}\n")

        for i in range(min(n_images, figures)):
            plt.imshow(tf.reshape(spsas[i], (28, 28)), cmap='gist_gray')
            plt.title(f"{data_type} image (bay)")
            plt.savefig(f"./images_mnist/{data_type}_{i}_bay_image.jpeg", bbox_inches='tight')
            plt.clf()
            plt.bar(x, means[i])
            plt.title("prediction (bay)")
            plt.ylim([0, 1])
            plt.savefig(f"./images_mnist/{data_type}_{i}_bay_prediction.jpeg", bbox_inches='tight')
            plt.clf()

        return res


    n_images = 128
    figures = 10
    thresholds = [0.0, 0.3, 0.5]
    epsilons = [0.3]

    for e in epsilons:
        det_batch_spsa(n_images=n_images, figures=figures, thresholds=thresholds, epsilon=e)

    for e in epsilons:
        bay_batch_spsa_fb(n_images=n_images, figures=figures, thresholds=thresholds, epsilon=e)





if __name__ == "__main__": main()