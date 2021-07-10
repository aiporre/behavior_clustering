import argparse
import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)
# try:
#     # Disable all GPUS
#     tf.config.set_visible_devices([], 'GPU')
#     visible_devices = tf.config.get_visible_devices()
#     for device in visible_devices:
#         assert device.device_type != 'GPU'
# except:
#   # Invalid device or cannot modify virtual devices once initialized.
#   print('CANNOT DISABLE GPUs')
#   pass


from losses import OPWMetric, triplet_loss
from models.pose_embedding import PoseEmbeddings
from datasets import create_dataset
from time import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
#tf.debugging.set_log_device_placement(True)

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

def main(dataset_path, dataset_name, saved_model_name, verbose, plotting, plot_samples=None, epochs=10, min_distance=None, save_iters=100, batch_size = 16, max_opts=5, binary=False):
    if binary:
        dataset_1 = create_dataset(dataset_name, dataset_path=dataset_path, with_labels=False, shuffle=True, binary=True).build()
        dataset_2 = create_dataset(dataset_name, dataset_path=dataset_path, with_labels=False, shuffle=True, binary=True).build()
    else:
        dataset_1 = create_dataset(dataset_name, dataset_path=dataset_path, with_labels=False, shuffle=True, binary=False).parallelize_extraction().build()
        dataset_2 = create_dataset(dataset_name, dataset_path=dataset_path, with_labels=False, shuffle=True, binary=False).parallelize_extraction().build()
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
    t_epochs = trange(epochs, desc='Training running loss: --.--e--', leave=True)

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

    # Distance threshold
    if min_distance is None:
        print('Estimating of minimal distance. This can take a while')
        distances = []
        for s1,s2 in tqdm(dataset.take(20), total=20):
            if compare_sequences(s1, s2):
                print('Samples equal...')
                continue
            # num_seq_1, num_seq_2 = s1.shape[0], s2.shape[0]
            # s1 = s1[:min(num_seq_1, batch_size*max_opts)]
            # s2 = s2[:min(num_seq_2, batch_size*max_opts)]
            sequences_phi = [model.predict(s1, batch_size=8), model.predict(s2, batch_size=8)]
            distance, transport = opw_metric(sequences_phi[0], sequences_phi[1])
            distances.append(distance)
        if plotting:
            plt.figure()
            plt.hist(distances)
            plt.title('Distribution of distances extracted')
            plt.show()
        min_distance = 0.9 * max(distances)
        print('New min_distance is now: ', min_distance)


    for e in t_epochs:
        t_dataset = trange(dataset_1.length, desc='Training running loss: --.--e--', leave=True)
        print('Epoch ', e)
        losses = 0
        cnt = 0
        save_iters_cnt = 0
        for sequence_1, sequence_2 in dataset:
            cnt += 1
            print('processing seg cnt=', cnt)
            # TODO: use tf.equal all or smth like that. Do not use numpy()
            # Force reduction of frames
            # num_seq_1, num_seq_2 = sequence_1.shape[0], sequence_2.shape[0]
            # sequence_1 = sequence_1[:min(num_seq_1, batch_size*max_opts)]
            # sequence_2 = sequence_2[:min(num_seq_2, batch_size*max_opts)]
            # print(sequence_1.shape)
            # print(sequence_2.shape)

            if compare_sequences(sequence_1, sequence_2):
                print('Samples equal...')
                continue

            if verbose:
                print('Computing all pose embeddings to sample pose pairs...')
            timer.start()
            # RQ: => Is it possible to optimize transport together with the embedding?
            # RQ: => Can be the mining be part of the optimization? Seems totally like a stupid question but maybe not.
            # FIXME: batch size is hardcoded!
            sequence_1_phi, sequence_2_phi = (model.predict(sequence_1, batch_size=8), model.predict(sequence_2, batch_size=8))
            if verbose:
                timer.lap()
                print('Computing optimal transport...')
            # samples = [d[0], d[1]]
            # distance, transport = opw_metric(samples[0].reshape((N,-1)), samples[1].reshape((M,-1)))
            # FIXME: poses are in an array?? Why?!

            distance, transport = opw_metric(sequence_1_phi, sequence_2_phi)
            min_distance = 0.9 * min_distance + 0.3 * distance
            if distance > min_distance:
                print('Condition not fulfilled > min_distance :', distance, '>', min_distance)
                continue
            # TODO: implement print_verbose function as print_verbose("hello world :) ") as in https://stackoverflow.com/questions/5980042/how-to-implement-the-verbose-or-v-option-into-a-script
            if verbose:
                print('OPW metric distance: ',  distance)
                timer.lap()
                print('Computing positive, and negative pairs')
            num_seq_1, num_seq_2 = sequence_1.shape[0], sequence_2.shape[0]
            # seq_1_x = sequences[0]
            # seq_2_x = sequences[1]
            # Create anchors
            # anchors = sequences[0]
            # Positive samples
            positive_assigment = np.argmax(transport, axis=1)
            # positive_samples = tf.gather_nd(sequences[1], [[a] for a in positive_assigment])
            # Negative samples
            mid = np.sqrt(1 / num_seq_1 ** 2 + 1 / num_seq_2 ** 2)
            ii, jj = np.mgrid[1:num_seq_1 + 1, 1:num_seq_2 + 1]
            distance_matrix = np.random.rand(num_seq_1, num_seq_2)+0.3*np.abs(ii / num_seq_1 - jj / num_seq_2)
            #distance_matrix = np.zeros([num_seq_1, num_seq_2])

            # FIXME: seq_distance should be also include disimilarty/distance
            #def seq_distance(x, y, i, j):
                # np.linalg.norm(x - y)
                # return np.random.rand() + 0.3 * abs(i - j)
            # FIXME: op is not vectorized!
            for i in range(num_seq_1):
                for j in range(num_seq_2):
                    if positive_assigment[i] == j:
                        distance_matrix[i, j] = 0
            #        else:
            #            distance_matrix[i, j] = seq_distance(sequences[0][i], sequences[1][j], i / num_seq_1, j / num_seq_2)


            negative_assignment = np.argmax(distance_matrix, axis=1)
            # negative_samples = tf.gather_nd(sequences[1], [[a] for a in negative_assignment])
            # FIXME: distance calculation are repeated? maybe using distance matrix is useful
            d_p = np.linalg.norm(sequence_1_phi - sequence_2_phi[positive_assigment], axis=1)
            d_n = np.linalg.norm(sequence_1_phi - sequence_2_phi[negative_assignment], axis=1)
            # Creating the semi-hard_mask
            # FIXME: as in https://arxiv.org/abs/1503.03832 if no semi-hard sample use the largest negative dist neg sample
            # Creating the hard_mask
            hard_mask = (d_p < d_n) * (d_n < d_p + margin_f)
            hard_indices = [i for i, m in enumerate(hard_mask) if m]
            if len(hard_indices) == 0:
                if verbose:
                    print('no semi-hard samples has been found, skipping...')
                continue
            hard_indices = np.random.choice(hard_indices, size=batch_size*max_opts)
            positive_assigment = positive_assigment[hard_indices]
            negative_assignment = negative_assignment[hard_indices]
            anchors = tf.gather(sequence_1, hard_indices)
            positive_samples = tf.gather(sequence_2, positive_assigment)
            negative_samples = tf.gather(sequence_2, negative_assignment)
            if verbose:
                timer.lap()
                print("number of hard triples = ", sum(hard_mask), ' out of ', len(hard_mask))
                print('anchor len           : ', anchors.shape, type(anchors))
                print('positive_samples len : ', positive_samples.shape, type(positive_samples))
                print('negative_samples len : ', negative_samples.shape, type(negative_samples))
                #print('POSITIVE-assing :', positive_assigment)
                #:xprint('NEGATIVE-assing :', negative_assignment)
                print('Optimizing on triplets...')

            if plotting:
                cnt_plot = 0
                S_anchors = anchors.numpy()
                S_prima = sequence_1.numpy()
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
            current_index = 0
            L = 0
            cnt2 = 0
            if batch_size > anchors.shape[0]:
                if verbose:
                    print('=> skipping not enough samples for training < batchsize')
                continue

            while current_index + batch_size <= anchors.shape[0]:
                cnt2 += 1
                L += train_step(anchors[current_index: current_index + batch_size],
                                positive_samples[current_index: current_index + batch_size],
                                negative_samples[current_index: current_index + batch_size]).numpy()
                current_index += batch_size
                t_dataset.set_description('Training running loss B: {:e}'.format(L / cnt2))
            if not cnt2==0:
                loss = L / cnt2
            else:
                loss = np.nan
            if not np.isnan(loss):
                losses += loss
            else:
                print('! loss is NAN ')
            t_dataset.set_description('Training running loss: {:e}'.format(losses / cnt))

            # saving after save iters
            save_iters_cnt += cnt2
            if save_iters_cnt > save_iters:
                model.save_weights(save_model_path)
                if verbose:
                    print('Model saved after: ', save_iters_cnt, ' optimization steps.')
                save_iters_cnt = 0

            if verbose:
                timer.stop()
            t_dataset.update(1)

        print('epoch loss: {:e}'.format(losses / cnt))
        print('saving model: ', save_model_path)
        model.save_weights(save_model_path)
        t_epochs.set_description('Training running loss: {:e}'.format(losses / cnt))
        t_epochs.update(1)
        t_dataset.close()
    t_epochs.close()

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
    parser.add_argument('--min-dist', metavar='min-dist', default=None, type=float,
                        help="minimal distance to consider sequence within the curriculum learning")
    parser.add_argument('--save-iters', metavar='save-iters', default=100, type=int,
                        help="number of iterations to save weights")
    parser.add_argument('--batch-size', metavar='batch-size', default=16, type=int,
                        help="Batch size of optimization steps")
    parser.add_argument('--max-opts', metavar='max-opts', default=5, type=int,
                        help="Maximal optimization steps")

    parser.add_argument('--binary', action="store_true",
                        help="Sets dataset to create/load a binaries files. Meaning that it creates binary files from the orignal dataset.")

    args = parser.parse_args()
    print(args)
    main(dataset_path=args.dataset_path, dataset_name=args.dataset_name, saved_model_name=args.saved_model, verbose=args.verbose,
         plotting=args.plotting, plot_samples=args.plot_samples, epochs=args.epochs, min_distance = args.min_dist, save_iters=args.save_iters, batch_size=args.batch_size, max_opts=args.max_opts, binary=args.binary)
