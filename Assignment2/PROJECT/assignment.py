import os
import random
import sys

from matplotlib import cm

import util
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, silhouette_score, silhouette_samples, adjusted_rand_score
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold


def load_dataset(path):
    df = pd.read_csv(path)
    # TODO: exception handling, e.g. data has not been cleaned? -> instruct to run setup.py

    return df


def identify_n_clusters(X, y):
    # elbow method
    ssqdist = []
    range_n_clusters = range(1, 7)
    for k in range_n_clusters:
        km = KMeans(n_clusters=k)
        km = km.fit(X)
        ssqdist.append(km.inertia_)

    # display plot to user, showing
    plt.plot(range_n_clusters, ssqdist, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Enter ideal k in console...')
    plt.show()

    # prompt user to enter number of clusters
    print('Enter number of clusters identified from plots (integer): ')
    k_elbow =  int(input())

    # Find k which gives highest silhouette score

    k_silhouette = 0
    max_silhouette_score = 0

    range_n_clusters = range(2, 7)
    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        # print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

        if silhouette_avg > max_silhouette_score:
            max_silhouette_score = silhouette_avg
            k_silhouette = n_clusters

    max_cluster_score = 0
    # range of possible k values has been identified:
    print("best cluster number identified by silhouette method: ", k_silhouette)

    for k in range(min(k_silhouette, k_elbow), max(k_silhouette, k_elbow) + 1):

        km = KMeans(n_clusters=k).fit(X)
        labels = km.predict(X)
        score = adjusted_rand_score(labels, y);

        print('cluster score', k, score)

        if score > max_cluster_score:
            max_cluster_score = score
            best_clustering = km

    return km, k


def data_report():
    df_dir = "data/"

    # dataset file names
    datasets = ["breast_cancer_data.csv", "wine_data.csv", "weather_aus_data.csv"]

    final_results = []

    for filename in datasets:
        df = load_dataset(os.path.join(df_dir, filename))

        X = df.drop(columns=['class']).values
        y = df['class'].to_numpy()

        util.print_separator()

        print('Dataset: ', filename)

        skf = StratifiedKFold(n_splits=10, shuffle=True)

        permutation_index = 1

        scores = []
        # For each of the permutations:

        for (train_index, test_index) in skf.split(X, y):
            util.print_separator()
            print("Partition permutation: ", permutation_index)

            X_train = X[train_index]
            y_train = y[train_index]
            X_test = X[test_index]
            y_test = y[test_index]

            # identify minority class
            minority_label = np.argmin(np.bincount(y_train))
            print('classes in train: ', np.bincount(y_train))
            print('minority label: ', minority_label)

            #
            km, k = identify_n_clusters(X_train, y_train)

            # 5. For each cluster, identify centroid and number of samples of the minority class in that cluster
            y_pred = km.predict(X_train)

            cluster_forests = []

            for i in range(0, k):
                print('Cluster ', i)
                print('Centroid: ', km.cluster_centers_[i])
                print('Minority samples: ', np.sum(np.logical_and(y_train == minority_label, y_pred == i)))

                # 6. Train a random forest for each of the clusters that contains samples from more than one class (
                # if a cluster only has samples for one of the classes, you don’t need to train a classifier).
                # Find class distribution of cluster
                cluster_distribution = np.bincount(y_train[y_pred == i])
                print('Cluster class distribution: ', cluster_distribution)

                if cluster_distribution.size > 1:
                    classifier = RandomForestClassifier().fit(X_train, y_train)
                    cluster_forests.append(classifier)
                else:
                    cluster_forests.append(None)

            # 7. Given a sample x from the unseen fold (the one left out in (3))
            # Assign x to its closest cluster.
            # If this cluster has only instances of one class, assign to x that label. Otherwise, use the model
            # trained with data from that cluster to assign a label to x.

            y_cluster = km.predict(X_test)
            y_pred = []

            for xi, yi in zip(X_test, y_cluster):
                if cluster_forests[yi] is None:
                    y_pred.append(yi)
                else:
                    # must reshape data point xi into 2D array
                    y_pred.append(cluster_forests[yi].predict(xi.reshape(1, -1))[0])

            score = f1_score(y_pred, y_test, average='weighted')

            scores.append(score)
            permutation_index += 1

        util.print_separator()

        print('Final scoring for dataset')
        print('Mean score: ', np.mean(scores))
        print('Standard deviation: ', np.std(scores))

        final_results.append(scores)

        util.print_separator()

    # Creates three subplots
    fig, axes = plt.subplots(1, 3, sharey=True)

    for r, ax, lb in zip(final_results, axes, ['Cancer', 'Wine', 'Weather']):
        ax.boxplot(r , labels=[lb])

    plt.title('Proposed Solution')
    plt.xlabel('Dataset')
    plt.ylabel('F1 score')
    plt.show()


if __name__ == "__main__":
    data_report()
