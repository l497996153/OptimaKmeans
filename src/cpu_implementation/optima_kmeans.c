#include "OptimaKmeans/optima_kmeans_cpu.h"
#include "dataloader.h"
#include "kmeans.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


int optima_load_data_bin(const char* filename, float** data, int* n, int* d) {
    return load_data_bin(filename, data, n, d);
}

int optima_load_data_csv(const char* filename, float** data, int* n, int* d) {
    return load_data_csv(filename, data, n, d);
}

float* optima_kmeans(float *points, int num_points, int dim, int k, int max_iter, int *clusters, int *iter_converge) {
    return kmeans(points, num_points, dim, k, max_iter, clusters, iter_converge);
}

void optima_malloc_clusters(int** clusters, int n) {
    *clusters = malloc(n * sizeof(int));
}

void optima_free_data(float* data, float* centroids, int* clusters) {
    free_data(data);
    free(centroids);
    free(clusters);
}
