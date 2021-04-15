import argparse
import tensorflow as tf
from losses import OPWMetric, triplet_loss
from models.pose_embedding import PoseEmbeddings
from datasets import create_dataset
from time import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


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

def compare_sequences(seq1, seq2):
    return False

def main(dataset_path, dataset_name, saved_model_name, verbose, plotting, plot_samples=None, epochs=10):
    dataset_1 = create_dataset(dataset_name, dataset_path=dataset_path, with_labels=False, shuffle=True).build()
    dataset_2 = create_dataset(dataset_name, dataset_path=dataset_path, with_labels=False, shuffle=True).build()
    dataset = tf.data.Dataset.zip((dataset_1, dataset_2))
    opw_metric = OPWMetric(lambda_1=150, lambda_2=0.5)
    model = PoseEmbeddings(image_size=(100, 100), use_l2_normalization=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    margin_f = 0.1
    margin = tf.constant(0.1)
    lossfcn = triplet_loss
    timer = Timer()
    save_model_path = f'saved_models/{saved_model_name}'
    try:
        model.load_weights(save_model_path)
        print('Model loaded')
    except:
        print('Model is not loaded')
    t = trange(epochs, desc='Training running loss: --.--e--', leave=True)

    # TODO: use keras iteration_step instead.. maybe?
    @tf.function
    def train_step(_anchors, _positive_samples, _negative_samples):
        with tf.GradientTape() as tape:
            x_a = model(_anchors)
            x_p = model(_positive_samples)
            x_n = model(_negative_samples)
            loss = lossfcn(x_a, x_p, x_n, margin=margin)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss

    for e in t:
        print('Epoch ', e)
        # Distance threshold
        min_distance = 100.0
        losses = 0
        cnt = 0
        for sequences in dataset:
            cnt += 1
            # TODO: use tf.equal all or smth like that. Do not use numpy()
            if False: #compare_sequences(sequences[0], sequences[1]):
                print('Samples equal...')
                continue

            if verbose:
                print('Computing all pose embeddings to sample pose pairs...')
            timer.start()
            # RQ: => Is it possible to optimize transport together with the embedding?
            # RQ: => Can be the mining be part of the optimization? Seems totally like a stupid question but maybe not.
            # FIXME: batch size is hardcoded!
            sequences_phi = [model.predict(sequences[0], batch_size=8), model.predict(sequences[1], batch_size=8)]
            if verbose:
                timer.lap()
                print('Computing optimal transport...')
            # samples = [d[0], d[1]]
            # distance, transport = opw_metric(samples[0].reshape((N,-1)), samples[1].reshape((M,-1)))
            # FIXME: poses are in an array?? Why?!
            distance, transport = opw_metric(sequences_phi[0], sequences_phi[1])

            if distance > min_distance:
                print('Condition not fulfilled > min_distance :', distance, '>', min_distance)
                continue
            # TODO: implement print_verbose function as print_verbose("hello world :) ") as in https://stackoverflow.com/questions/5980042/how-to-implement-the-verbose-or-v-option-into-a-script
            if verbose:
                print('OPW metric distance: ',  distance)
                timer.lap()
                print('Computing positive, and negative pairs')
            num_seq_1, num_seq_2 = sequences[0].shape[0], sequences[1].shape[0]
            # seq_1_x = sequences[0]
            # seq_2_x = sequences[1]
            # Create anchors
            anchors = sequences[0]
            # Positive samples
            positive_assigment = np.argmax(transport, axis=1)
            positive_samples = tf.gather_nd(sequences[1], [[a] for a in positive_assigment])
            # Negative samples
            distance_matrix = np.zeros([num_seq_1, num_seq_2])

            # FIXME: seq_distance should be also include disimilarty/distance
            def seq_distance(x, y, i, j):
                # np.linalg.norm(x - y)
                return np.random.rand() + 0.3 * abs(i - j)
            # FIXME: op is not vectorized!
            for i in range(num_seq_1):
                for j in range(num_seq_2):
                    if positive_assigment[i] == j:
                        distance_matrix[i, j] = 0
                    else:
                        distance_matrix[i, j] = seq_distance(sequences[0][i], sequences[1][j], i / num_seq_1, j / num_seq_2)

            negative_assignment = np.argmax(distance_matrix, axis=1)
            negative_samples = tf.gather_nd(sequences[1], [[a] for a in negative_assignment])
            # FIXME: distance calculation are repeated? maybe using distance matrix is useful
            d_p = np.linalg.norm(sequences_phi[0] - sequences_phi[1][positive_assigment], axis=1)
            d_n = np.linalg.norm(sequences_phi[0] - sequences_phi[1][negative_assignment], axis=1)
            # Creating the semi-hard_mask
            # FIXME: as in https://arxiv.org/abs/1503.03832 if no semi-hard sample use the largest negative dist neg sample
            # Creating the hard_mask
            hard_mask = (d_p < d_n) * (d_n < d_p + margin_f)
            anchors = anchors[hard_mask]
            positive_samples = positive_samples[hard_mask]
            negative_samples = negative_samples[hard_mask]
            if verbose:
                timer.lap()
                print("number of hard triples = ", sum(hard_mask), ' out of ', len(hard_mask))
                print('anchor len           : ', anchors.shape, type(anchors))
                print('positive_samples len : ', positive_samples.shape, type(positive_samples))
                print('negative_samples len : ', negative_samples.shape, type(negative_samples))
                # print('POSITIVE-assing :', positive_assigment)
                # print('NEGATIVE-assing :', negative_assignment)
                print('Optimizing on triplets...')

            if plotting:
                cnt_plot = 0
                S_anchors = anchors.numpy()
                S_prima = sequences[1].numpy()
                for orginal, assignment_p, assignment_n in zip(S_anchors, positive_assigment, negative_assignment):
                    cnt_plot += 1
                    plt.figure(figsize=(10, 5))
                    plt.subplot(1, 3, 1)
                    plt.title(f'original sample 1 #{cnt_plot}')
                    plt.imshow(orginal)
                    plt.subplot(1, 3, 2)
                    plt.imshow(S_prima[assignment_p])
                    plt.title(f'(+) sample1 #{cnt_plot} on sample 2 #{assignment_p}')
                    plt.subplot(1, 3, 3)
                    plt.imshow(S_prima[assignment_n])
                    plt.title(f'(-) sample1 #{cnt_plot} on sample 2 #{assignment_n}')
                    plt.show()
                    if plot_samples is not None and cnt_plot >= plot_samples:
                        break
            # FIXME: batch size is hardcoded!!
            batch_size = 16
            current_index = 0
            L = 0
            cnt2 = 0
            while current_index < anchors.shape[0]:
                cnt2 += 1
                L += train_step(anchors[current_index: current_index + batch_size],
                                positive_samples[current_index: current_index + batch_size],
                                negative_samples[current_index: current_index + batch_size]).numpy()
                current_index += batch_size
                t.set_description('Training running loss B: {:e}'.format(L / cnt2))
            if not cnt2==0:
                loss = L / cnt2
            else:
                loss = np.nan
            if not np.isnan(loss):
                losses += loss
            else:
                print('! loss is NAN ')
            t.set_description('Training running loss: {:e}'.format(losses / cnt))
            if verbose:
                timer.stop()
        print('epoch loss: {:e}'.format(losses / cnt))
        print('saving model: ', save_model_path)
        model.save_weights(save_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dp", "--dataset-path", metavar='path_to_videos',
                        type=str,
                        help='Path to the image files')
    parser.add_argument('-dn', "--dataset-name", metavar='name_of_dataset', default='mitsinglemouse',
                        help='Name of the dataset. Options: \'mitsinglemouse\' , \'olympicsports\'')
    parser.add_argument("-sm", "--saved-model", metavar='some_name_of_your_model',
                        type=str,
                        help='Name of the model to be load/saved into folder saved_models.')
    parser.add_argument("-v", "--verbose", action='store_true',
                        help='Name of the model to be load/saved into folder saved_models.')
    parser.add_argument("-p", "--plotting", action='store_true',
                        help="Activate plotting input triplets")
    parser.add_argument("-p:n", "--plot-samples", metavar='number_of_samples', type=int, default=None,
                        help="Limits the number of samples to plot")
    parser.add_argument('-e', "--epochs", type=int, default=10,
                        help='Number of epochs to train')
    args = parser.parse_args()
    print(args)
    main(dataset_path=args.dataset_path, dataset_name=args.dataset_name, saved_model_name=args.saved_model, verbose=args.verbose,
         plotting=args.plotting, plot_samples=args.plot_samples, epochs=args.epochs)
