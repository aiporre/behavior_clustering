from pathlib import Path

from dpipe.utils import get_video_length, get_read_fcn
from dpipe import make_dataset, from_function
from os import path
from random import shuffle as shuffle_list
from skimage.transform import resize

label_maps = {"drink" : 0,
    "d" : 0,
    "eat" : 1,
    "e" : 1,
    "groomback" : 2,
    "groom" : 2,
    "gb" : 2,
    "g" : 2,
    "hang" : 3,
    "ha" : 3,
    "micromovement" : 4,
    "head" : 4,
    "he" : 4,
    "rear": 5,
    "r" : 5,
    "rest" : 6,
    "rs" : 6,
    "walk" : 7,
    "w" : 7
}
label_maps_unique = {"drink" : 0,
    "eat" : 1,
    "groom" : 2,
    "hang" : 3,
    "micromovement" : 4,
    "rear": 5,
    "rest" : 6,
    "walk" : 7,
}


# def read_sample(sample_path):
#     read_video_fcn = get_read_fcn('video')
#     sample = read_video_fcn(sample_path).astype('float32')
#     return (sample-sample.min())/(sample.max()-sample.min())


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def read_sample(sample_path):
    read_video_fcn = get_read_fcn('video')
    sample = read_video_fcn(sample_path).astype('float32')
    length = sample.shape[0]
    sample = resize(sample, (length, 240, 320, 3), anti_aliasing=True)
    return normalize(sample)

def create_dataset(dataset_path, with_labels=False, shuffle=False):
    files = list(map(lambda x: x.as_posix(), Path(dataset_path).rglob('*.mpg')))
    if shuffle:
        shuffle_list(files)
    if with_labels:
        labels = list(map(lambda x: label_maps[path.basename(x).split('_')[-2]], files))
        def read_sample_with_labels(inputs):
            path, label = inputs
            video = read_sample(path)
            return video, label
        files_with_labels = list(zip(files, labels))
        dataset = from_function(read_sample_with_labels, files_with_labels, undetermined_shape=[[0],[]])
    else:
        dataset = from_function(read_sample, files, undetermined_shape=[0])
    return dataset

