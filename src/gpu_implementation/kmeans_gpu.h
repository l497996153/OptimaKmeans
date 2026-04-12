#ifndef KMEANS_GPU_H
#define KMEANS_GPU_H

#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

// Finding the nearest centroid for every point
__global__ void find_centroid(double *d_data, double *d_centroids, int *d_clusters, 
                                  int N, int D, int K);

// Calculating the centroids
__global__ void calculate_centroid(double *d_data, int *d_clusters, double *d_new_centroids, 
                              int *d_counts, int N, int D, int K);

#endif