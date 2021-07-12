from pathlib import Path

from dpipe import from_function
from os import path
from random import shuffle as shuffle_list
from olympic_sports import read_seq
import numpy as np
from skimage.transform import resize
import os
from tqdm import tqdm
label_maps = {}

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def read_sample(sample_path):
    frames = read_seq(sample_path)
    lenght = len(frames)
    sample = np.array(frames, dtype=np.float32)
    sample = resize(sample, (lenght, 100, 100, 3), anti_aliasing=True)
    return normalize(sample)

def read_sample_npy(sample_path):
    frames = np.load(sample_path)
    lenght = len(frames)
    sample = np.array(frames, dtype=np.float32)
    sample = resize(sample, (lenght, 100, 100, 3), anti_aliasing=True)
    return normalize(sample)



def create_dataset(dataset_path, with_labels=False, shuffle=False, binary=False):
    if binary:
        assert not with_labels, 'Binary MIT single mouse dataset is default dataset. Make binary False'
    if binary:
        files = list(map(lambda x: x.as_posix(), Path(dataset_path).rglob('*.npy')))
    else:
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
    elif not binary:
        dataset = from_function(read_sample, files, undetermined_shape=[0])
    elif binary and len(files) == 0:
        # binary files weren't crated
        files = list(map(lambda x: x.as_posix(), Path(dataset_path).rglob('*.seq')))
        # new dataset path
        dataset_path_mod = os.path.join(dataset_path, 'binaries')
        if not os.path.exists(dataset_path_mod):
            os.mkdir(dataset_path_mod)
        print('Creating binary files for mouse dfeault dataset in ', dataset_path, ". This might take a while.")

        def write_sample(f):
            if isinstance(f, bytes):
                f = f.decode()
            try:
                value = read_sample(f)
                fname, _ = os.path.splitext(os.path.basename(f))
                inner_dir = Path(f).parent.as_posix().replace(dataset_path, "")
                inner_dir = inner_dir[1:] if inner_dir.startswith(os.path.sep) else inner_dir
                p_join = os.path.join(dataset_path_mod, inner_dir, fname + ".npy")
                if not os.path.exists(p_join):
                    os.makedirs(Path(p_join).parent, exist_ok=True)
                np.save(p_join, value)
            except Exception as e:
                print('Error in file ', f)
                pass
            return 1

        dataset = from_function(write_sample, files).parallelize_extraction()
        for value in tqdm(dataset.build().as_numpy_iterator(), desc="Creating binaries MitSingleDataset: ",total=len(files)):
            pass

        files_npy = list(map(lambda x: x.as_posix(), Path(dataset_path_mod).rglob('*.npy')))
        dataset = from_function(read_sample_npy, files_npy, undetermined_shape=[0])
    else:
        dataset = from_function(read_sample_npy, files, undetermined_shape=[0])

    return dataset
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ds = create_dataset(dataset_path='../data/human', with_labels=False).build()
    for d in ds:
        print(d.shape)
        plt.imshow(d[0])
        plt.show()