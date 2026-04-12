#include "kmeans_gpu.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


// Finding the nearest centroid for every point
__global__ void find_centroid(double *d_data, double *d_centroids, int *d_clusters, int N, int D, int K)
{
    //TODO
}

// Calculating the centroids
__global__ void calculate_centroid(double *d_data, int *d_clusters, double *d_new_centroids, int *d_counts, int N, int D, int K)
{
    //TODO
}

double* kmeans_gpu(double *data, int num_points, int dim, int k, int max_iteration, int *clusters)
{
    // TODO
    return NULL;
}