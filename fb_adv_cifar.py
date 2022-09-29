import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import datasets, layers, models, utils

tfd = tfp.distributions

import foolbox as fb
from foolbox import TensorFlowModel

import eagerpy as ep

import numpy as np

import matplotlib.pyplot as plt


def get_data_cifar():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    train_images = train_images.reshape(train_images.shape[0], 32, 32, 3)
    test_images = test_images.reshape(test_images.shape[0], 32, 32, 3)

    train_images, test_images = train_images / 255.0, test_images / 255.0

    train_images = tf.constant(train_images)
    test_images = tf.constant(test_images)

    train_labels = train_labels.reshape(train_labels.shape[0], )
    test_labels = test_labels.reshape(test_labels.shape[0], )

    train_labels = tf.cast(train_labels, tf.int32)
    test_labels = tf.cast(test_labels, tf.int32)

    return train_images, train_labels, test_images, test_labels


def get_targeted_label(x):
    if x == 0:
        return 8
    if x == 1:
        return 0
    if x == 2:
        return 3
    if x == 3:
        return 5
    if x == 4:
        return 7
    if x == 5:
        return 4
    if x == 6:
        return 2
    if x == 7:
        return 6
    if x == 8:
        return 9
    if x == 9:
        return 1


def main():

    train_images, train_labels, test_images, test_labels = get_data_cifar()

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

        model.load_weights('./model_weights/cifar_det').expect_partial()

        fmodel = TensorFlowModel(model, bounds=(0.0, 1.0), preprocessing=dict())

        return fmodel

    def create_model_bay():
        kl_divergence_fn = (lambda q, p, _: tfd.kl_divergence(q, p) / tf.cast(len(train_images), dtype=tf.float32))

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
        model.add(tfp.layers.DenseFlipout(10, kernel_divergence_fn=kl_divergence_fn, activation=tf.nn.softmax))

        #optimizer = tf.keras.optimizers.Adam(lr=0.001)
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=False)

        model.compile(optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'], experimental_run_tf_function=False)

        model.load_weights('./model_weights/cifar_bay').expect_partial()

        fmodel = TensorFlowModel(model, bounds=(0.0, 1.0), preprocessing=dict())

        return fmodel

    fmodel_det = create_model_det()
    fmodel_bay = create_model_bay()

    test_images, test_labels = ep.astensors(test_images, test_labels)

    def test_accuracy_det(data_type, images, labels, thresholds=[0.0], figures=0):
        n_images = len(images)

        print(f"\n\n\nDeterministic model accuracy with {data_type} data:\n")

        model_outputs = fmodel_det(images).raw.numpy()

        pred_max = np.max(model_outputs, axis=-1)
        predictions = np.argmax(model_outputs, axis=-1)

        res = []

        for i in range(len(thresholds)):
            acc = 0
            pred_correct = 0.0
            pred_incorrect = 0.0
            guessed = 0

            for j in range(n_images):
                if pred_max[j] >= thresholds[i]:
                    guessed += 1
                    if predictions[j] == labels[j]:
                        acc += 1
                        pred_correct += pred_max[j]
                    else:
                        pred_incorrect += pred_max[j]
            pred_correct = pred_correct / acc if acc > 0 else -1
            pred_incorrect = pred_incorrect / (guessed - acc) if guessed - acc > 0 else -1
            acc = acc / guessed if guessed > 0 else -1

            res.append((thresholds[i], acc, guessed / n_images, pred_correct, pred_incorrect))

            if thresholds[i] == 0.0:
                print(f"Accuracy using no confidence threshold:\t\t\t\t{res[i][1] * 100:4.1f}%")
            else:
                print(f"Accuracy using a confidence threshold of {res[i][0]}:\t\t\t{res[i][1] * 100:4.1f}%"
                      f"\t({res[i][2] * 100:4.1f}% of images used)")
            print(f"Prediction confidence among correctly classified images:\t{res[i][3]:4.3f}")
            print(f"Prediction confidence among incorrectly classified images:\t{res[i][4]:4.3f}\n")

        for i in range(min(n_images, figures)):
            x = ['plane', 'auto', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck']
            plt.imshow(images[i].raw.numpy().reshape(32, 32, 3))
            plt.title(f"{data_type} image (det)")
            plt.savefig(f"./images_cifar/{data_type}_{i}_det_image.jpeg", bbox_inches='tight')
            plt.clf()
            plt.bar(x, model_outputs[i])
            plt.title("prediction (det)")
            plt.ylim([0, 1])
            plt.savefig(f"./images_cifar/{data_type}_{i}_det_prediction.jpeg", bbox_inches='tight')
            plt.clf()

        return res

    def test_accuracy_bay(data_type, images, labels, thresholds=[0.0], figures=0, iterations=100):
        n_images = len(images)

        print(f"\n\n\nBayesian model accuracy with {data_type} data:\n")

        model_outputs = []

        for i in range(iterations):
            model_outputs.append(fmodel_bay(images).raw.numpy())

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
                    if predictions[j] == labels[j]:
                        acc += 1
                        mean_correct += mean_max[j]
                    else:
                        mean_incorrect += mean_max[j]

            mean_correct = mean_correct/acc if acc > 0 else -1
            mean_incorrect = mean_incorrect/(guessed-acc) if guessed - acc > 0 else -1
            acc = acc / guessed if guessed > 0 else -1

            res.append((thresholds[i], acc, guessed/n_images, mean_correct, mean_incorrect))

            if thresholds[i] == 0.0:
                print(f"Accuracy using no confidence threshold:\t\t\t\t{res[i][1] * 100:4.1f}%")
            else:
                print(f"Accuracy using a confidence threshold of {res[i][0]}:\t\t\t{res[i][1] * 100:4.1f}%"
                      f"\t({res[i][2] * 100:4.1f}% of images used)")
            print(f"Prediction confidence among correctly classified images:\t{res[i][3]:4.3f}")
            print(f"Prediction confidence among incorrectly classified images:\t{res[i][4]:4.3f}\n")

        for i in range(min(n_images, figures)):
            x = ['plane', 'auto', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck']
            plt.imshow(images[i].raw.numpy().reshape(32, 32, 3))
            plt.title(f"{data_type} image (bay)")
            plt.savefig(f"./images_cifar/{data_type}_{i}_bay_image.jpeg", bbox_inches='tight')
            plt.clf()
            plt.bar(x, means[i])
            plt.title("prediction (bay)")
            plt.ylim([0, 1])
            plt.savefig(f"./images_cifar/{data_type}_{i}_bay_prediction.jpeg", bbox_inches='tight')
            plt.clf()

        return res

    def gen_attack(fmodel, images, labels, epsilons):
        print(f"\n\nStarted genetic attack")

        names = []
        for eps in epsilons:
            names.append(f"genetic{eps}")

        attack = fb.attacks.GenAttack()

        labels = labels.raw.numpy()

        for i in range(len(labels)):
            labels[i] = get_targeted_label(labels[i])

        labels = ep.astensors(tf.constant(labels))
        labels = labels[0]
        criterion = fb.criteria.TargetedMisclassification(labels)

        raw_advs, clipped_advs, success = attack(fmodel, images, criterion, epsilons=epsilons)

        return names, clipped_advs

    def carliniwagner_attack(fmodel, images, labels, epsilons):
        print(f"\n\nStarted carlini wagner attack")

        names = []
        for eps in epsilons:
            names.append(f"carliniwagner{eps}")

        attack = fb.attacks.L2CarliniWagnerAttack()

        criterion = fb.criteria.Misclassification(labels)

        attack.__init__(binary_search_steps=12, steps=500, stepsize=1, initial_const=1, confidence=0.5)

        _, clipped_advs, _ = attack(fmodel, images.float32(), criterion, epsilons=epsilons)

        return names, clipped_advs

    def fgm_attack(fmodel, images, labels, epsilons):
        print(f"\n\nStarted fast gradient method attack")

        names = []
        for eps in epsilons:
            names.append(f"fgm{eps}")

        attack = fb.attacks.L2FastGradientAttack()

        criterion = fb.criteria.Misclassification(labels)

        _, clipped_advs, _ = attack(fmodel, images.float32(), criterion, epsilons=epsilons)

        return names, clipped_advs


    n_images = 128

    n_figures = 10

    thresholds = [0.0, 0.3, 0.5]
    epsilons = [0.05, 0.1]

    images = test_images[:n_images]
    labels = test_labels[:n_images]

    #test_accuracy_det("clean", images, labels, thresholds, n_figures)
    #test_accuracy_bay("clean", images, labels, thresholds, n_figures)

    names, adv_images = gen_attack(fmodel_det, images, labels, epsilons)
    for i in range(len(adv_images)):
        test_accuracy_det(names[i], adv_images[i], test_labels, thresholds, n_figures)

    names, adv_images = gen_attack(fmodel_bay, images, labels, epsilons)
    for i in range(len(adv_images)):
        test_accuracy_bay(names[i], adv_images[i], test_labels, thresholds, n_figures)



if __name__ == "__main__": main()