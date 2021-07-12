import argparse
from models.pose_embedding import PoseEmbeddings
from datasets import create_dataset, get_label_map
import umap
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def compare_sequences(seq1, seq2):
    return False

def main(dataset_path, dataset_name, saved_model_name, verbose, plotting, plot_samples=None, epochs=10, min_distance=None):
    dataset = create_dataset(dataset_name, dataset_path=dataset_path, with_labels=True, shuffle=True).build()
    label_maps = get_label_map(dataset_name)
    model = PoseEmbeddings(image_size=(100, 100), use_l2_normalization=True)
    model.build(input_shape=(None,100,100,3))
    save_model_path = f'saved_models/{saved_model_name}'
    try:
        model.load_weights(save_model_path)
        print('Model loaded')
    except Exception as e:
        print(e)
        print('Model is not loaded')

    pose_df = pd.DataFrame()
    cnt = 0
    N = 5
    for sequence, label in tqdm(dataset.take(N).as_numpy_iterator(), total=N):
        cnt += 1
        sequences_phi = model.predict(sequence, batch_size=8)
        for phi in sequences_phi:
            inputs = {'label':label}
            for i in range(len(phi)):
                inputs['phi_'+str(i)] = phi[i]
            new_row = pd.DataFrame(inputs, index=[0])
            pose_df = pd.concat([pose_df, new_row], ignore_index=True)
    reducer = umap.UMAP()
    scaled_pose_df = StandardScaler().fit_transform(pose_df.drop(columns=['label']).values)
    embedding = reducer.fit_transform(scaled_pose_df)
    embedding_df = pd.DataFrame({"p_x": embedding[:,0], "p_y": embedding[:,1], "label": pose_df['label']})
    invered_label_maps = {v: k for k, v in label_maps.items()}
    embedding_df['label_name'] = embedding_df.label.map(invered_label_maps)
    plt.figure()
    plt.gca().set_aspect('equal', 'datalim')
    sns.scatterplot(data=embedding_df, x='p_x', y='p_y', hue='label_name')
    plt.title('UMAP projection of the pose dataset', fontsize=12)
    plt.show()


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
    args = parser.parse_args()
    print(args)
    main(dataset_path=args.dataset_path, dataset_name=args.dataset_name, saved_model_name=args.saved_model, verbose=args.verbose,
         plotting=args.plotting, plot_samples=args.plot_samples, epochs=args.epochs, min_distance = args.min_dist)
