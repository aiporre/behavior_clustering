import argparse
from datasets import mit_single_mouse_read_sample
from losses import OPWMetric
import matplotlib.pyplot as plt

def main(file_names, plotting):
    sample_1 = mit_single_mouse_read_sample(file_names[0])
    sample_2 = mit_single_mouse_read_sample(file_names[1])
    sample_1_flatten = sample_1.reshape((sample_1.shape[0],-1))
    sample_2_flatten = sample_2.reshape((sample_2.shape[0], -1))
    metric = OPWMetric(maxIter=100)
    # plotting sample_1 to vs its assigments on sample_2
    if plotting:
        assign_1, assign_2 = metric.calculate_assigment(sample_1_flatten, sample_2_flatten, only_indices=True)
        cnt = 0
        for orginal, assignment in zip(sample_1, assign_1):
            cnt += 1
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.title(f'original sample 1 {cnt}')
            plt.imshow(orginal)
            plt.subplot(1, 2, 2)
            plt.imshow(sample_2[assignment])
            plt.title(f'assigment of sample1 #{cnt} on sample 2 #{assignment}')
            plt.show()
    else:
        d, T = metric(sample_1_flatten, sample_2_flatten)
        print('DISTANCE: ', d)
        print('TRANSPORT: ', T)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-fn", "--file-names", metavar='path_to_file',
                        type=str,
                        nargs=2,
                        help='pair of files to containing the sequencial data inputs.')
    parser.add_argument("--plot", action='store_true',
                        help='plot image of assignments')
    args = parser.parse_args()
    print(args)
    main(args.file_names, args.plot)
