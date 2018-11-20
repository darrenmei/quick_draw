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
    prediction = np.zeros((test.shape[0]))

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

    return prediction

def compute_performance(prediction, test, num_clusters):
    classifiers = test.iloc[:,test.shape[1] - num_clusters: test.shape[1]]
    num_samples = test.shape[0]
    num_positive_results = np.zeros(num_clusters)
    num_correct = np.zeros(num_clusters)
    num_relevant = np.zeros(num_clusters)

    for i in range(test.shape[0]):
        num_positive_results[int(prediction[i])] += 1
        if classifiers.iloc[i][int(prediction[i])] == 1:
            num_correct[int(prediction[i])] += 1

    success_rates = np.zeros(num_clusters)
    f1_scores = np.zeros(num_clusters)
    for j in range(num_clusters):
        num_relevant[j] = classifiers.iloc[:,j].sum()
        precision = num_correct[j] / num_positive_results[j]
        recall = num_correct[j] / num_relevant[j]
        success_rates[j] = recall
        f1_scores[j] = (2 * (precision) * (recall)) / (precision + recall)

    return success_rates, f1_scores

def main():
    num_clusters = 5

    train_file = 'npy_data/train_npy.csv'
    train = pd.read_csv(train_file)

    test_file = 'npy_data/test_npy.csv'
    test = pd.read_csv(test_file)

    dev_file = 'npy_data/dev_npy.csv'
    dev = pd.read_csv(dev_file)

    centroids = compute_centroids(train, num_clusters)

    dev_prediction = classify_points(centroids, dev, num_clusters)
    dev_success_rate, dev_f1 = compute_performance(dev_prediction, dev, num_clusters)
    print('Dev Success Rates:', dev_success_rate)
    print('Dev Average Success Rate:', np.mean(dev_success_rate))
    print('Dev F1 Scores:', dev_f1)
    print('Dev Average F1 Score:', np.mean(dev_f1))

    test_prediction = classify_points(centroids, test, num_clusters)
    test_success_rate, test_f1 = compute_performance(test_prediction, test, num_clusters)
    print('Test Success Rates:', test_success_rate)
    print('Test Average Success Rate:', np.mean(test_success_rate))
    print('Test F1 Scores:', test_f1)
    print('Test Average F1 Score:', np.mean(test_f1))

if __name__ == '__main__':
	main()
