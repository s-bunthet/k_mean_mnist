import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import time
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser(description="Arguments to train K_Mean Algorithm.")
parser.add_argument('--vb', default=0, type=int, choices=[0, 1], help='Verbosity mode of the training.0 without verbose and 1 with verbose')
parser.add_argument('--init', type=str, default='k-means++', choices=['k-means++', 'random', 'cheat'], help='The way to choose initials centroids.')
parser.add_argument('--n', type=int, default=10, help='Number of clusters.')
parser.add_argument('--n-init', type=int, default=10, help='Number of time the k-means will be run with different centroid seeds.')
parser.add_argument('--n-jobs', type=int, default=2, help='Number of jobs for parallel computation. Choose -1 to use all available processors.')
parser.add_argument('--algo', type=str, default='auto', choices=['auto', 'full', 'elkan'], help='K-mean algorithm to use.')
parser.add_argument('--max-iter', type=int, default=300, help='Maximum number of iterations of the k-means algorithm for a single run.')


def plot_confusion_matrix(confusion_matrix):
    """
    Plot a confusion matrix.
    :param confusion_matrix:
    :return:
    """
    labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix)

    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_xlabel("predicted labels")
    ax.set_ylabel("real labels")

    for i in range(10):
        for j in range(10):
            text = ax.text(j, i, confusion_matrix[i, j], ha="center", va="center", color="w")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    plt.show()


def train():
    args = parser.parse_args()
    # load data
    train_data = pd.read_csv('data/optdigits.tra').values

    test_data = pd.read_csv('data/optdigits.tes').values
    if args.init != 'cheat':
        k_mean = KMeans(n_clusters=args.n, init=args.init, n_init=args.n_init,
                        verbose=args.vb, n_jobs=args.n_jobs, algorithm=args.algo,
                        max_iter=args.max_iter)
    else:
        # In 'cheat' mode, we will initialize the clusters' centroid base on the labels of the train_data.
        # Basically, we look for 10 feature-array in train_data that each array correspond to each labels.
        # This might increase the performance of the algorithm compare the mode 'k-mean++' and 'random' since
        # the performance of K-mean is really sensitive to the initialized centroids.

        init_centroids = []
        labels_array = np.arange(10)
        for i in range(train_data.shape[0]):
            if train_data[i, -1] in labels_array:
                init_centroids.append(train_data[i, np.arange(train_data.shape[1]-1)])
                labels_array = np.delete(labels_array, np.argwhere(labels_array == train_data[i, -1]))
                if labels_array.shape[0] == 0:
                    break
        k_mean = KMeans(n_clusters=args.n, init=np.array(init_centroids),
                        verbose=args.vb, n_jobs=args.n_jobs, algorithm=args.algo,
                        max_iter=args.max_iter)

    start_train_time = time.time()
    k_mean = k_mean.fit(train_data[:, np.arange(train_data.shape[1]-1)])
    print('Training time: {0:.4f} seconds.'.format(time.time()-start_train_time))
    labels = k_mean.labels_
    dic = {}  # to map from labels given by algorithm and the real label by doing one-shot encoding
    for i in range(train_data.shape[0]):
        dic.update({labels[i]: train_data[i, -1]})
        if np.unique(list(dic.values())).shape[0] == args.n:  # make sure that all values in 'labels' is map with different real labels from the train_data
            break
    # replace the labels done k_mean with real label
    predicted_labels = k_mean.predict(test_data[:, np.arange(test_data.shape[1]-1)])
    # confusion matrix
    start_prediction_time = time.time()
    confusion_matrix = np.zeros((10, 10))
    for i in range(test_data.shape[0]):
        confusion_matrix[test_data[i, -1]][dic[predicted_labels[i]]] += 1
    print("Arguments used for training: ", args.__dict__)
    print('Prediction time: {0:.4f} seconds'.format(time.time() - start_prediction_time))
    # Accuracy
    acc = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    print("Accuracy: {0:.4f}".format(acc))
    plot_confusion_matrix(confusion_matrix)


if __name__ == "__main__":
    train()


