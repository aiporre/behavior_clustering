import argparse
from datasets import mit_single_mouse_create_dataset
import tensorflow as tf
from losses import OPWMetric
from models.pose_embedding import PoseEmbeddings
from time import time
import numpy as np
from itertools import product
from sklearn.cluster import SpectralClustering


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
        print('Time passed: ' + str(self.stop_time-self.lap_time))
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
    opw_metric = OPWMetric()
    model = PoseEmbeddings(image_size=(100, 100))
    timer = Timer()
    for e in range(epochs):
        print('Epoch ', e)
        print('Computing all pose embeddings to sample pose pairs...')
        poses = []
        timer.start()
        for d in dataset:
            poses.append(model.predict(d))
            timer.lap()
        timer.stop()
        print('Computing distance matrix and finding best candidate')
        timer.start()
        num_poses_seq = len(poses)
        min_distance = np.inf
        best_pair = None
        for i,j in product(range(num_poses_seq), range(num_poses_seq)):
            distance, transport = opw_metric(poses[i], poses[j])
            if distance<min_distance:
                best_pair = (i,j)
        timer.stop()


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
