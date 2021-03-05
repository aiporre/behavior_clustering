from dpipe.utils import get_video_length, get_read_fcn


def read_sample(sample_path):
    read_video_fcn = get_read_fcn('video')
    sample = read_video_fcn(sample_path).astype(float)
    return sample/sample.max()
