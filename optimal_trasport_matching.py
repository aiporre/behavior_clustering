import argparse
from datasets import mit_single_mouse_read_sample, olympic_read_sample
from losses import OPWMetric
import matplotlib.pyplot as plt

from models.pose_embedding import PoseEmbeddings

available_dataset = {'mitsinglemouse': mit_single_mouse_read_sample,
                     'olympicsports': olympic_read_sample}

def main(file_names, plotting, model_path, image_based, dataset_name):
    sample_1 = available_dataset[dataset_name](file_names[0])
    sample_2 = available_dataset[dataset_name](file_names[1])
    if image_based:
        sample_1_flatten = sample_1.reshape((sample_1.shape[0],-1))
        sample_2_flatten = sample_2.reshape((sample_2.shape[0], -1))
    else:
        model = PoseEmbeddings(image_size=(100, 100), use_l2_normalization=True)
        try:
            save_model_path = f'saved_models/{model_path}'
            model.load_weights(save_model_path)
            print('Model loaded')
        except:
            print('Model is not loaded' )
        sample_1_flatten = model.predict(sample_1)
        sample_2_flatten = model.predict(sample_2)


    metric = OPWMetric(lambda_1=150, lambda_2=0.5)

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
    parser.add_argument('-dn', "--dataset-name", metavar='name_of_dataset', default='mitsinglemouse',
                        help='Name of the dataset. Options: \'mitsinglemouse\' , \'olympicsports\'')
    parser.add_argument("--plot", action='store_true',
                        help='plot image of assignments')
    parser.add_argument("--model-path", '-m', type=str, default='saved_models/mouse/',
                        help='Model to load path')
    parser.add_argument('--image-based', '-im',  action='store_true',
                        help='Calculates assignment from images as flatten vectors, otherwise uses a pose embedding. The model will initialized if the if model is given')
    args = parser.parse_args()
    print(args)
    main(args.file_names, args.plot, args.model_path, args.image_based, args.dataset_name)
