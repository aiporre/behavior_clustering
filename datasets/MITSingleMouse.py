import os.path
from pathlib import Path

import numpy as np
from dpipe.utils import get_video_length, get_read_fcn
from dpipe import make_dataset, from_function
from os import path
from random import shuffle as shuffle_list
from skimage.transform import resize
from tqdm import tqdm

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
    length = min(sample.shape[0], 256)
    sample = resize(sample, (length, 224, 224, 3), anti_aliasing=True)
    return normalize(sample)


def read_sample_npy(sample_path):
    return np.load(sample_path)

def create_dataset(dataset_path, with_labels=False, shuffle=False, binary=False):
    if binary:
        assert not with_labels, 'Binary MIT single mouse dataset is default dataset. Make binary False'
    if binary:
        files = list(map(lambda x: x.as_posix(), Path(dataset_path).rglob('*.npy')))
    else:
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
    elif not binary:
        dataset = from_function(read_sample, files, undetermined_shape=[0])
    elif binary and len(files) == 0:
        # binary files weren't crated
        files = list(map(lambda x: x.as_posix(), Path(dataset_path).rglob('*.mpg')))
        # new dataset path
        dataset_path_mod = os.path.join(dataset_path, 'binaries')
        if not os.path.exists(dataset_path_mod):
            os.mkdir(dataset_path_mod)
        print('Creating binary files for mouse dfeault dataset in ', dataset_path, ". This might take a while.")
        def write_sample(f):
            if isinstance(f, bytes):
                f = f.decode()
            value = read_sample(f)
            fname, _ = os.path.splitext(os.path.basename(f))
            inner_dir = Path(f).parent.as_posix().replace(dataset_path, "")
            inner_dir = inner_dir[1:] if inner_dir.startswith(os.path.sep) else inner_dir
            p_join = os.path.join(dataset_path_mod, inner_dir, fname + ".npy")
            if not os.path.exists(p_join):
                os.makedirs(Path(p_join).parent, exist_ok=True)
            np.save(p_join, value)
            return 1
        dataset = from_function(write_sample, files).parallelize_extraction()
        for value in tqdm(dataset.build().as_numpy_iterator(), desc="Creating binaries MitSingleDataset: ", total=len(files)):
            pass

        files_npy = list(map(lambda x: x.as_posix(), Path(dataset_path_mod).rglob('*.npy')))
        dataset = from_function(read_sample_npy, files_npy, undetermined_shape=[0])
    else:
        dataset = from_function(read_sample_npy, files, undetermined_shape=[0])
    return dataset

