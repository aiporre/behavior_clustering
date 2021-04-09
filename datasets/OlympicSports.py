from pathlib import Path

from dpipe import from_function
from os import path
from random import shuffle as shuffle_list
from olympic_sports import read_seq
import numpy as np
from skimage.transform import resize

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def read_sample(sample_path):
    frames = read_seq(sample_path)
    lenght = len(frames)
    sample = np.array(frames, dtype=np.float32)
    sample = resize(sample, (lenght, 100, 100, 3), anti_aliasing=True)
    return normalize(sample)


def create_dataset(dataset_path, with_labels=False, shuffle=False):
    files = list(map(lambda x: x.as_posix(), Path(dataset_path).rglob('*.seq')))
    if shuffle:
        shuffle_list(files)
    if with_labels:
        labels = list(map(lambda x: path.basename(path.dirname(x)), files))
        print(labels)
        def read_sample_with_labels(inputs):
            path, label = inputs
            video = read_sample(path)
            return video, label
        files_with_labels = list(zip(files, labels))
        dataset = from_function(read_sample_with_labels, files_with_labels, undetermined_shape=[[0],[]])
    else:
        dataset = from_function(read_sample, files, undetermined_shape=[0])
    return dataset

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ds = create_dataset(dataset_path='../data/human', with_labels=False).build()
    for d in ds:
        print(d.shape)
        plt.imshow(d[0])
        plt.show()