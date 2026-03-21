#ifndef KMEANS_H
#define KMEANS_H

// A simple structure to hold a data point and its assigned cluster
typedef struct {
    double *features;  // The feature vector of this point
    int cluster;     // The cluster ID assigned to this point
} Point;

/**
 * @param points: Array of Point structs
 * @param num_points: Total points in the dataset
 * @param dim: Number of dimensions (features) per point
 * @param k: Number of clusters to find
 * @param max_iteration: Maximum iterations to run the algorithm
 * @return: A 2D array of the final centroids [k][dim]
 */
double** kmeans(Point *points, int num_points, int dim, int k, int max_iter);

#endif