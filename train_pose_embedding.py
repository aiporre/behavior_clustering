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


def main(dataset_path, dataset_name, saved_model_name, verbose, plotting, plot_samples=None, epochs=10):
    dataset_1 = create_dataset(dataset_name, dataset_path=dataset_path, with_labels=False, shuffle=True).build()
    dataset_2 = create_dataset(dataset_name, dataset_path=dataset_path, with_labels=False, shuffle=True).build()
    dataset = tf.data.Dataset.zip((dataset_1, dataset_2))
    opw_metric = OPWMetric(lambda_1=150, lambda_2=0.5)
    model = PoseEmbeddings(image_size=(100, 100), use_l2_normalization=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
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

    @tf.function(experimental_relax_shapes=True)
    def train_step(anchors, positive_samples, negative_samples):
        with tf.GradientTape() as tape:
            x_a = model(anchors)
            x_p = model(positive_samples)
            x_n = model(negative_samples)
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
        for d in dataset:
            cnt += 1
            if d[0].shape[0] == d[1].shape[0] and (d[0] == d[1]).numpy().all():
                print('Samples equal...')
                continue
            poses = []
            if verbose:
                print('Computing all pose embeddings to sample pose pairs...')
            timer.start()
            poses.append(model.predict(d[0], batch_size=8))
            poses.append(model.predict(d[1], batch_size=8))
            if verbose:
                timer.lap()
                print('Computing optimal transport...')
            # samples = [d[0], d[1]]
            # distance, transport = opw_metric(samples[0].reshape((N,-1)), samples[1].reshape((M,-1)))
            distance, transport = opw_metric(poses[0], poses[1])

            if distance > min_distance:
                print('Condition not fulfilled > min_distance :', distance, '>', min_distance)
                continue
            if verbose:
                timer.lap()
                print('Computing positive, and negative pairs')
            seq_1 = poses[0]
            seq_2 = poses[1]
            num_seq_1, num_seq_2 = seq_1.shape[0], seq_2.shape[0]
            seq_1_x = d[0]
            seq_2_x = d[1]
            # Create anchors
            anchors = seq_1_x
            # Positive samples
            positive_assigment = np.argmax(transport, axis=1)
            positive_samples = tf.gather_nd(seq_2_x, [[a] for a in positive_assigment])
            # Negative samples
            distance_matrix = np.zeros([num_seq_1, num_seq_2])

            def seq_distance(x, y, i, j):
                # np.linalg.norm(x - y)
                return np.random.rand() + 0.3 * abs(i - j)

            for i in range(num_seq_1):
                for j in range(num_seq_2):
                    if positive_assigment[i] == j:
                        distance_matrix[i, j] = 0
                    else:
                        distance_matrix[i, j] = seq_distance(seq_1[i], seq_2[j], i / num_seq_1, j / num_seq_2)

            negative_assignment = np.argmax(distance_matrix, axis=1)
            negative_samples = tf.gather_nd(seq_2_x, [[a] for a in negative_assignment])
            d_p = np.linalg.norm(seq_1 - seq_2[positive_assigment], axis=1)
            d_n = np.linalg.norm(seq_1 - seq_2[negative_assignment], axis=1)
            # Creating the hard_mask
            hard_mask = (d_p < d_n) * (d_n < d_p + 1)
            anchors = anchors[hard_mask]
            positive_samples = positive_samples[hard_mask]
            negative_samples = negative_samples[hard_mask]
            if verbose:
                timer.lap()
                print("number of hard triples = ", sum(hard_mask), ' out of ', len(hard_mask))
                print('anchor len           : ', anchors.shape, type(anchors))
                print('positive_samples len : ', positive_samples.shape, type(positive_samples))
                print('negative_samples len : ', negative_samples.shape, type(negative_samples))
                print('Optimizing on triplets...')

            if plotting:
                cnt_plot = 0
                S_anchors = anchors.numpy()
                S_prima = seq_2_x.numpy()
                for orginal, assignment_p, assignment_n in zip(S_anchors, positive_assigment, negative_assignment):
                    cnt_plot += 1
                    plt.figure(figsize=(10, 5))
                    plt.subplot(1, 3, 1)
                    plt.title(f'original sample 1 #{cnt}')
                    plt.imshow(orginal)
                    plt.subplot(1, 3, 2)
                    plt.imshow(S_prima[assignment_p])
                    plt.title(f'(+) sample1 #{cnt} on sample 2 #{assignment_p}')
                    plt.subplot(1, 3, 3)
                    plt.imshow(S_prima[assignment_n])
                    plt.title(f'(-) sample1 #{cnt} on sample 2 #{assignment_n}')
                    plt.show()
                    if plot_samples is not None and cnt_plot >= plot_samples:
                        break
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
            loss = L / cnt2
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
