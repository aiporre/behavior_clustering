import argparse
from datasets import mit_single_mouse_create_dataset
import tensorflow as tf
from losses import OPWMetric, triplet_loss
from models.pose_embedding import PoseEmbeddings
from time import time
import numpy as np
from itertools import product
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt

class Timer(object):
    def __init__(self):
        self.start_time = 0
        self.stop_time = 0
        self.lap_time = 0

    def start(self):
        self.start_time = time()
        self.stop_time = self.start_time
        self.lap_time = self.start_time

    def lap(self):
        self.stop_time = time()
        print('Time passed: ' + str(self.stop_time - self.lap_time))
        self.lap_time = self.stop_time

    def stop(self):
        self.stop_time = time()
        print('Total time passed: ' + str(self.stop_time - self.start_time))
        self.start_time = self.stop_time
        self.lap_time = self.stop_time


def main(dataset_path, epochs=10):
    N = 10
    dataset = mit_single_mouse_create_dataset(dataset_path, with_labels=False).build().take(N)
    # dataset = tf.data.Dataset.zip((dataset, dataset)).take(20)
    dataset.length = N
    opw_metric = OPWMetric(lambda_1=150, lambda_2=0.5)
    model = PoseEmbeddings(image_size=(100, 100))
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-1)

    timer = Timer()
    for e in range(epochs):
        print('Epoch ', e)
        print('Computing all pose embeddings to sample pose pairs...')
        poses = []
        samples = []
        timer.start()
        for d in dataset:
            poses.append(model.predict(d))
            samples.append(d.numpy())
        timer.stop()
        print('Computing distance matrix and finding best candidate')
        timer.start()
        num_poses_seq = len(poses)
        min_distance = np.inf
        best_pair = None
        best_transport = None
        for i, j in product(range(num_poses_seq), range(num_poses_seq)):
            if i == j:
                continue
            distance, transport = opw_metric(poses[i], poses[j])
            if distance < min_distance:
                best_pair = (i, j)
                best_transport = transport
        timer.stop()

        print('Computing distance matrix and finding best candidate')

        seq_1 = poses[best_pair[0]]
        seq_2 = poses[best_pair[1]]
        num_seq_1 = seq_1.shape[0]
        num_seq_2 = seq_2.shape[0]


        print('Computing positive, and negative pairs')
        print(best_pair)
        seq_1_x = samples[best_pair[0]]
        seq_2_x = samples[best_pair[1]]
        print('seq_2_x.shape==> ', seq_2_x.shape)
        anchors = seq_1_x
        positive_assigment = np.argmax(best_transport, axis=1)
        print('positive sampling: ', positive_assigment)
        positive_samples = seq_2_x[positive_assigment]

        distance_matrix = np.zeros([num_seq_1, num_seq_2])
        def seq_distance(x, y, i, j):
            #np.linalg.norm(x - y)
            return np.random.rand() + 0.3 * abs(i - j)

        for i in range(num_seq_1):
            for j in range(num_seq_2):
                if positive_assigment[i] == j:
                    distance_matrix[i, j] = 0
                else:
                    distance_matrix[i, j] = seq_distance(seq_1[i], seq_2[j], i / num_seq_1, j / num_seq_2)

        negative_assignment = np.argmax(distance_matrix, axis=1)
        print('negative sampling: ', negative_assignment)
        negative_samples = seq_2_x[negative_assignment]
        # print('anchors to positive: ', np.linalg.norm(anchors - positive_samples))
        # print('anchors to negative: ', np.linalg.norm(anchors - negative_samples))
        #
        print('anchor len           : ', anchors.shape)
        print('positive_samples len : ', positive_samples.shape)
        print('negative_samples len : ', negative_samples.shape)
        print('Optimizing on triplets...')

        # cnt=0
        # for orginal, assignment in zip(anchors, positive_assigment):
        #     cnt += 1
        #     plt.figure()
        #     plt.subplot(1, 2, 1)
        #     plt.title(f'original sample 1 #{cnt}')
        #     plt.imshow(orginal)
        #     plt.subplot(1, 2, 2)
        #     plt.imshow(seq_2_x[assignment])
        #     plt.title(f'assigment of sample1 #{cnt} on sample 2 #{assignment}')
        #     plt.show()
        with tf.GradientTape() as tape:
            x_a = model(anchors)
            x_p = model(positive_samples)
            x_n = model(negative_samples)
            loss = triplet_loss(x_a, x_p, x_n, margin=1.0)
        print('loss: ', float(loss))
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))




        # print('Clustering sequences..')
        # timer.start()
        # gamma = 1.0
        # adjacency_matrix = np.exp(-gamma * distance_matrix ** 2)
        # sc = SpectralClustering(3, affinity='precomputed', n_init=100,
        #                         assign_labels='discretize')
        # labels = sc.fit_predict(adjacency_matrix)
        # print(labels)
        # timer.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dp", "--dataset-path", metavar='path_to_videos',
                        type=str,
                        help='Path to the image files')
    args = parser.parse_args()
    print(args)
    main(args.dataset_path)
