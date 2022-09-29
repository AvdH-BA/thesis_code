from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
from tensorflow.keras import datasets, layers, models, utils

import statistics
import matplotlib.pyplot as plt

from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.tf2.attacks.momentum_iterative_method import momentum_iterative_method
from cleverhans.tf2.attacks.spsa import spsa

tfd = tfp.distributions
tfpl = tfp.layers

x = ['bicycle', 'bird']

def get_data_bob():
    test_dataset = utils.image_dataset_from_directory('./bob_dataset/test', label_mode='binary',
                                                      image_size=(32, 32))

    normalization_layer = layers.Rescaling(1./255)
    test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

    test_dataset = tfds.as_numpy(test_dataset)

    test_images = np.zeros((300, 32, 32, 3))
    test_labels = np.zeros(300)
    i = 0

    for element in test_dataset:
        j = len(element[0])
        test_images[i:i+j] = element[0]
        test_labels[i:i+j] = element[1].reshape(element[1].shape[0], )
        i += j

    test_images = test_images.reshape(test_images.shape[0], 32, 32, 3)
    test_images = tf.constant(test_images)

    test_labels = tf.cast(test_labels, tf.int32)

    test_labels = utils.to_categorical(test_labels)

    return test_images, test_labels

class BOBSequence(tf.keras.utils.Sequence):

    def __init__(self, data=None, batch_size=128):
        if data:
            images, labels = data
        else:
            images, labels = BOBSequence.__generate_fake_data(
                num_images=128, num_classes=2)
        self.images, self.labels = BOBSequence.__preprocessing(images, labels)
        self.batch_size = batch_size

    @staticmethod
    def __preprocessing(images, labels):
        return images, labels

    def __generate_fake_data(num_images, num_classes):
        images = np.random.randint(low=0, high=256,
                                   size=(num_images, 32, 32, 3))
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

    batch_size = 128

    test_i, test_l = get_data_bob()

    heldout_seq = BOBSequence(data=(test_i, test_l), batch_size=batch_size)

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

        optimizer = tf.keras.optimizers.Adam(lr=0.001)

        model.compile(optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

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

        #optimizer = tf.keras.optimizers.Adam(lr=0.001)
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=False)

        model.compile(optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'], experimental_run_tf_function=False)

        return model

    model_det = create_model_det()
    model_bay = create_model_bay()

    model_det.load_weights('./model_weights/bob_det_sm').expect_partial()
    model_bay.load_weights('./model_weights/bob_bay_sm').expect_partial()


    def det_batch_spsa(batch_number=0, n_images=batch_size, thresholds=[0.0], figures=0, epsilon=0.1):
        test_images, test_labels = BOBSequence.__getitem__(heldout_seq, batch_number)

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
            plt.imshow(tf.reshape(spsas[i], (32, 32, 3)))
            plt.title(f"{data_type} image (det)")
            plt.savefig(f"./images_bob/{data_type}_{i}_det_image.jpeg", bbox_inches='tight')
            plt.clf()
            plt.bar(x, pred[0])
            plt.title("prediction (det)")
            plt.ylim([0, 1])
            plt.savefig(f"./images_bob/{data_type}_{i}_prediction.jpeg", bbox_inches='tight')
            plt.clf()

        return res

    def bay_batch_clean(batch_number=0, n_images=batch_size, n_rep=100):
        test_images, test_labels = BOBSequence.__getitem__(heldout_seq, batch_number)

        prob_bay_batch = np.zeros((n_rep, n_images, 10))

        for i in range(n_rep):
            predictions = model_bay.predict_on_batch(test_images)
            for j in range(n_images):
                for k in range(10):
                    prob_bay_batch[i][j][k] = predictions[j][k]

        prob_bay_batch_median = np.zeros((n_images, 10))
        prob_bay_batch_threshold = np.zeros((n_images, 10))

        for k in range(10):
            for i in range(n_images):
                list = []
                for j in range(n_rep):
                    list.append(prob_bay_batch[j][i][k])
                median = statistics.median(list)
                prob_bay_batch_median[i][k] = median
                if median > 0.2:
                    prob_bay_batch_threshold[i][k] = 1

        n_wrong = 0.0
        n_guessed = 0.0

        for i in range(n_images):
            summed = np.sum(prob_bay_batch_threshold[i])
            if summed == 1:
                n_guessed += 1
                guess = np.argmax(prob_bay_batch_threshold[i])
                actual = np.argmax(test_labels[i])
                if guess != actual:
                    n_wrong += 1

        perc_abstained = (n_images - n_guessed) / n_images

        accuracy = -1
        if perc_abstained < 1:
            accuracy = (n_guessed - n_wrong) / n_guessed

        return accuracy, perc_abstained


    def bay_batch_spsa_fb(batch_number=0, n_images=batch_size, thresholds=[0.0], figures=0, iterations=100, epsilon=0.1):
        data_type = f"spsa{epsilon}"

        test_images, test_labels = BOBSequence.__getitem__(heldout_seq, batch_number)

        x_32 = tf.cast(test_images, tf.float32)

        model_outputs = np.zeros((iterations, n_images, 2))

        spsas = []

        for i in range(n_images):
            x_spsa = spsa(model_fn=model_bay, x=x_32[i][None, :, :, :], y=test_labels[i], spsa_iters=1, delta=0.01,
                          eps=epsilon, nb_iter=100, clip_max=1.0, clip_min=-1.0, learning_rate=0.5, is_debug=False)
            spsas.append(x_spsa)

        for i in range(iterations):
            for j in range(n_images):
                prediction = model_bay.predict(spsas[j], verbose=0)
                for k in range(2):
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
                print(f"\n\n\nDeterministic model accuracy with {data_type} data:\n")

            if thresholds[i] == 0.0:
                print(f"Accuracy using no confidence threshold:\t\t\t\t{res[i][1] * 100:4.1f}%")
            else:
                print(f"Accuracy using a confidence threshold of {res[i][0]}:\t\t\t{res[i][1] * 100:4.1f}%"
                      f"\t({res[i][2] * 100:4.1f}% of images used)")
            print(f"Prediction confidence among correctly classified images:\t{res[i][3]:4.3f}")
            print(f"Prediction confidence among incorrectly classified images:\t{res[i][4]:4.3f}\n")

        for i in range(min(n_images, figures)):
            plt.imshow(tf.reshape(spsas[i], (32, 32, 3)))
            plt.title(f"{data_type} image (bay)")
            plt.savefig(f"./images_bob/{data_type}_{i}_bay_image.jpeg", bbox_inches='tight')
            plt.clf()
            plt.bar(x, means[i])
            plt.title("prediction (bay)")
            plt.ylim([0, 1])
            plt.savefig(f"./images_bob/{data_type}_{i}_bay_prediction.jpeg", bbox_inches='tight')
            plt.clf()

        return res

    n_images = 128
    figures = 10
    thresholds = [0.0, 0.65, 0.8]
    epsilons = [0.05]

    for e in epsilons:
        det_batch_spsa(n_images=n_images, figures=figures, thresholds=thresholds, epsilon=e)

    for e in epsilons:
        bay_batch_spsa_fb(n_images=n_images, figures=figures, thresholds=thresholds, epsilon=e)






if __name__ == "__main__": main()