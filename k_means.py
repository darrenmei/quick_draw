import numpy as np
import pandas as pd
import sys

def compute_centroids(train, num_clusters):
    centroids = np.zeros((num_clusters,train.shape[1] - num_clusters))
    classifiers = train.iloc[:,train.shape[1] - num_clusters: train.shape[1]]
    counter = np.zeros((num_clusters))

    for i in range(train.shape[0]):
        for j in range(num_clusters):
            if classifiers.iloc[i,j] == 1:
                counter[j] += 1
                centroids[j] += train.iloc[i, 0:train.shape[1] - 5]
    for j in range(num_clusters):
        centroids[j,:] /= counter[j]

    return centroids

def classify_points(centroids, test, num_clusters):
    classifiers = test.iloc[:,test.shape[1] - num_clusters: test.shape[1]]
    prediction = np.zeros((test.shape[0]))
    num_correct = 0

    for i in range(test.shape[0]):
        closest_dist = 1e10
        closest_centroid = -1
        for j in range(num_clusters):
            if np.linalg.norm(centroids[j] - test.iloc[i,0:test.shape[1] - 5]) < closest_dist:
                closest_centroid = j
                closest_dist = np.linalg.norm(centroids[j] - test.iloc[i,0:test.shape[1] - 5])
        prediction[i] = closest_centroid
        if prediction[i] == -1:
            print('Closest Centroid not identified')

    for i in range(test.shape[0]):
        if classifiers.iloc[i][int(prediction[i])] == 1:
            num_correct += 1

    return num_correct / test.shape[0]

def main():
    num_clusters = 5

    train_file = 'npy_data/train_npy.csv'
    train = pd.read_csv(train_file)

    test_file = 'npy_data/test_npy.csv'
    test = pd.read_csv(test_file)

    dev_file = 'npy_data/dev_npy.csv'
    dev = pd.read_csv(dev_file)

    centroids = compute_centroids(train, num_clusters)

    dev_percentage = classify_points(centroids, dev, num_clusters)
    print('Dev Percentage:', dev_percentage)

    test_percentage = classify_points(centroids, test, num_clusters)
    print('Test Percentage:', test_percentage)

if __name__ == '__main__':
	main()
