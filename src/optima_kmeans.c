// Wrapper function to run k-means algorithm on a given dataset
#include "OptimaKmeans/optima_kmeans.h"
#include "dataloader.h"
#include "kmeans.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

float* kmeans_gpu_float(float *h_data, int num_points, int dim, int k, int max_iteration, int *h_clusters, int *finished_iterations);
float* kmeans_gpu_streaming_float(float *h_data, int num_points, int dim, int k, int max_iteration,
                                  int *h_clusters, int *finished_iterations);
int optima_load_data_bin(const char* filename, double** data, int* n, int* d) {
    return load_data_bin(filename, data, n, d);
}

int optima_load_data_csv(const char* filename, double** data, int* n, int* d) {
    return load_data_csv(filename, data, n, d);
}

double* optima_kmeans(double *points, int num_points, int dim, int k, int max_iter, int *clusters) {
    return kmeans(points, num_points, dim, k, max_iter, clusters);
}

void optima_malloc_clusters(int** clusters, int n) {
    *clusters = malloc(n * sizeof(int));
}

void optima_free_data(double* data, double* centroids, int* clusters) {
    if (data) free_data(data);
    if (centroids) free(centroids);
    if (clusters) free(clusters);
}

KMeansResult optima_kmeans_gpu(double *data, int num_points, int dim, int k, int max_iteration, int *clusters) {
    KMeansResult result;
    size_t total = (size_t)num_points * dim;
    float *data_f = (float *)malloc(total * sizeof(float));
    for (size_t i = 0; i < total; i++) {
        data_f[i] = (float)data[i];
    }

    float *centroids_f = kmeans_gpu_float(data_f, num_points, dim, k, max_iteration, clusters, &result.iterations);
    free(data_f);

    result.centroids = (double *)malloc((size_t)k * dim * sizeof(double));
    for (size_t i = 0; i < (size_t)k * dim; i++) {
        result.centroids[i] = (double)centroids_f[i];
    }
    free(centroids_f);
    return result;
}

KMeansResult optima_kmeans_gpu_streaming(double *data, int num_points, int dim, int k, int max_iteration, int *clusters) {
    KMeansResult result;
    size_t total = (size_t)num_points * dim;
    float *data_f = (float *)malloc(total * sizeof(float));
    for (size_t i = 0; i < total; i++) {
        data_f[i] = (float)data[i];
    }

    float *centroids_f = kmeans_gpu_streaming_float(data_f, num_points, dim, k, max_iteration, clusters, &result.iterations);
    free(data_f);

    result.centroids = (double *)malloc((size_t)k * dim * sizeof(double));
    for (size_t i = 0; i < (size_t)k * dim; i++) {
        result.centroids[i] = (double)centroids_f[i];
    }
    free(centroids_f);
    return result;
}