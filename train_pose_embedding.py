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
from tqdm import tqdm, trange


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





def main(dataset_path, epochs=1000):
    N = 2000
    dataset_1 = mit_single_mouse_create_dataset(dataset_path, with_labels=False, shuffle=True).shuffle(1).build().take(
        N).prefetch(1)
    dataset_2 = mit_single_mouse_create_dataset(dataset_path, with_labels=False, shuffle=True).shuffle(10).build().take(
        N).prefetch(1)
    dataset = tf.data.Dataset.zip((dataset_1, dataset_2))
    dataset.length = N
    opw_metric = OPWMetric(lambda_1=150, lambda_2=0.5)
    model = PoseEmbeddings(image_size=(100, 100), use_l2_normalization=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    margin = tf.constant(0.1)
    lossfcn = triplet_loss
    timer = Timer()
    save_model_path = 'saved_models/mouse/'
    try:
        model.load_weights(save_model_path)
        print('Model loaded')
    except:
        print('Model is not loaded')
    t = trange(epochs, desc='Training running loss: --.--e--', leave=True)

    @tf.function(experimental_relax_shapes=True)
    def train_step(anchors, positive_samples, negative_samples):
        with tf.GradientTape() as tape:
            x_a = model(anchors)
            x_p = model(positive_samples)
            x_n = model(negative_samples)
            loss = lossfcn(x_a, x_p, x_n, margin=margin)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss

    for e in t:
        print('Epoch ', e)
        # Distance threshold
        min_distance = 100.0
        losses = []
        for d in dataset:
            if d[0].shape[0] == d[1].shape[0] and (d[0] == d[1]).numpy().all():
                print('Samples equal...')
                continue
            poses = []
            samples = []
            # print('Computing all pose embeddings to sample pose pairs...')
            # timer.start()
            N, M = d[0].shape[0], d[1].shape[0]
            d = tf.concat([d[0], d[1]], axis=0)
            pose_pred = model.predict(d)
            poses += [pose_pred[:N], pose_pred[N:]]
            samples += [d[:N], d[N:]]
            # timer.lap()
            # print('Computing optimal transport...')
            # distance, transport = opw_metric(samples[0].reshape((N,-1)), samples[1].reshape((M,-1)))
            distance, transport = opw_metric(poses[0], poses[1])

            # timer.lap()
            if distance > min_distance:
                print('Condition not fulfilled > min_distance :', distance, '>', min_distance)
                continue
            # print('Computing positive, and negative pairs')
            seq_1 = poses[0]
            seq_2 = poses[1]
            num_seq_1, num_seq_2 = seq_1.shape[0], seq_2.shape[0]
            seq_1_x = samples[0]
            seq_2_x = samples[1]
            # print('seq_2_x.shape==> ', seq_2_x.shape)
            anchors = seq_1_x
            # Positive samples
            positive_assigment = np.argmax(transport, axis=1)
            # print('positive sampling: ', positive_assigment)
            positive_samples = tf.gather_nd(seq_2_x, [ [a] for a in positive_assigment])
            # Negative samples
            distance_matrix = np.zeros([num_seq_1, num_seq_2])

            def seq_distance(x, y, i, j):
                # np.linalg.norm(x - y)
                return np.random.rand() + 0.3 * abs(i - j)

            for i in range(num_seq_1):
                for j in range(num_seq_2):
                    if positive_assigment[i] == j:
                        distance_matrix[i, j] = 0
                    else:
                        distance_matrix[i, j] = seq_distance(seq_1[i], seq_2[j], i / num_seq_1, j / num_seq_2)

            negative_assignment = np.argmax(distance_matrix, axis=1)
            # print('negative sampling: ', negative_assignment)
            negative_samples = tf.gather_nd(seq_2_x, [ [a] for a in negative_assignment])
            d_p = np.linalg.norm(seq_1 - seq_2[positive_assigment], axis=1)
            d_n = np.linalg.norm(seq_1 - seq_2[negative_assignment], axis=1)
            # print('anchors to positive: ', d_p)
            # print('anchors to negative: ', d_n)
            # Ls = d_p - d_n + 1
            # print('Loss => ', Ls)
            # CC = d_p.shape[0]
            # print("number of easy triples = ", sum(d_p+1<d_n), ' out of ', CC)
            # print("number of semi-hard triples = ", sum(d_n<d_p), ' out of ', CC)
            hard_mask = (d_p < d_n) * (d_n < d_p + 1)
            # print("number of hard triples = ", sum(hard_mask), ' out of ', CC)
            #
            anchors = anchors[hard_mask]
            positive_samples = positive_samples[hard_mask]
            negative_samples = negative_samples[hard_mask]
            print('anchor len           : ', anchors.shape, type(anchors))
            print('positive_samples len : ', positive_samples.shape, type(positive_samples))
            print('negative_samples len : ', negative_samples.shape, type(negative_samples))
            print('Optimizing on triplets...')

            # cnt=0
            # for orginal, assignment_p, assignment_n in zip(anchors, positive_assigment, negative_assignment):
            #     cnt += 1
            #     plt.figure(figsize=(10,5))
            #     plt.subplot(1, 3, 1)
            #     plt.title(f'original sample 1 #{cnt}')
            #     plt.imshow(orginal)
            #     plt.subplot(1, 3, 2)
            #     plt.imshow(seq_2_x[assignment_p])
            #     plt.title(f'(+) sample1 #{cnt} on sample 2 #{assignment_p}')
            #     plt.subplot(1, 3, 3)
            #     plt.imshow(seq_2_x[assignment_n])
            #     plt.title(f'(-) sample1 #{cnt} on sample 2 #{assignment_n}')
            #     plt.show()
            batch_size = 16
            current_index = 0
            L = []
            while current_index < anchors.shape[0]:
                L.append(train_step(anchors[current_index: current_index + batch_size], positive_samples[current_index: current_index + batch_size], negative_samples[current_index: current_index + batch_size]).numpy())
                current_index +=batch_size
            loss = np.mean(L)
            if not np.isnan(loss):
                losses.append(loss)
            else:
                print('! loss is NAN ')
            t.set_description('Training running loss: {:e}'.format(np.mean(losses)))

            # print('Clustering sequences..')
            # timer.start()
            # gamma = 1.0
            # adjacency_matrix = np.exp(-gamma * distance_matrix ** 2)
            # sc = SpectralClustering(3, affinity='precomputed', n_init=100,
            #                         assign_labels='discretize')
            # labels = sc.fit_predict(adjacency_matrix)
            # print(labels)
            # timer.stop()
        print('epoch loss: {:e}'.format(np.mean(losses)))
        print('saving model: ', save_model_path)
        model.save_weights(save_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dp", "--dataset-path", metavar='path_to_videos',
                        type=str,
                        help='Path to the image files')
    args = parser.parse_args()
    print(args)
    main(args.dataset_path)
