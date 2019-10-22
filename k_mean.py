import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import time
from sklearn.cluster import KMeans, MiniBatchKMeans

parser = argparse.ArgumentParser(description="Arguments to train K_Mean Algorithm.")
parser.add_argument('--mode', type=str, default='batch', choices=['batch', 'mini_batch'], help='The training mode of k_mean.')
parser.add_argument('--vb', default=0, type=int, choices=[0, 1], help='Verbosity mode of the training.0 without verbose and 1 with verbose')
parser.add_argument('--init', type=str, default='k-means++', choices=['k-means++', 'random', 'manual'], help='The way to choose initials centroids.')
parser.add_argument('--n', type=int, default=10, help='Number of clusters.')
parser.add_argument('--n-init', type=int, default=10, help='Number of time the k-means will be run with different centroid seeds.')
parser.add_argument('--n-jobs', type=int, default=-1, help='Number of jobs for parallel computation. Choose -1 to use all available processors.')
parser.add_argument('--algo', type=str, default='auto', choices=['auto', 'full', 'elkan'], help='K-mean algorithm to use.')

# pd.read_table
# do scatter plot
# data.head()



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

    k_mean = KMeans(n_clusters=args.n, init=args.init, n_init=args.n_init,
                    verbose=args.vb, n_jobs=args.n_jobs, algorithm=args.algo)
    start_train_time = time.time()
    k_mean = k_mean.fit(train_data[:, np.arange(train_data.shape[1]-1)])
    print('Training time: {0:.4f} seconds.'.format(time.time()-start_train_time))
    labels = k_mean.labels_
    dic = {}  # to map from labels given by algorithm and the real label by doing one-shot encoding
    for i in range(train_data.shape[0]):
        dic.update({labels[i]: train_data[i, -1]})
        if np.unique(list(dic.values())).shape[0] == 10:  # make sure that all values in 'labels' is map with different real labels from the train_data
            break
    # replace the labels done k_mean with real label
    predicted_labels = k_mean.predict(test_data[:, np.arange(test_data.shape[1]-1)])
    # confusion matrix
    start_prediction_time = time.time()
    confusion_matrix = np.zeros((10, 10))
    for i in range(test_data.shape[0]):
        for j in np.arange(10):
            if j == dic[predicted_labels[i]]:
                confusion_matrix[j][dic[predicted_labels[i]]] += 1
    print("Arguments used for training: ", args.__dict__)
    print('Prediction time: {0:.4f} seconds'.format(time.time() - start_prediction_time))
    # Accuracy
    acc = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    print("Accuracy: {0:.4f}".format(acc))
    plot_confusion_matrix(confusion_matrix)


if __name__ == "__main__":
    train()


