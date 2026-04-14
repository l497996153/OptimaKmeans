// example main.c
#include <stdio.h>
#include <time.h>
#include "OptimaKmeans/optima_kmeans.h"

int main() {
    double* data;
    int n, d;

    // Load data from a csv file
    if (optima_load_data_csv("../data/final_processed.csv", &data, &n, &d) != 0) {
        fprintf(stderr, "Failed to load data from CSV file\n");
        return 1;
    }

    // Run k-means
    int k = 5; // number of clusters
    int max_iter = 10000;
    int* clusters;
    optima_malloc_clusters(&clusters, n);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    KMeansResult gpu_result = optima_kmeans_gpu(data, n, d, k, max_iter, clusters);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed_sec = (double)(t1.tv_sec - t0.tv_sec) +
                         (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;
    printf("k-means elapsed time: %.6f s (%.3f ms)\n", elapsed_sec, elapsed_sec * 1e3);
    printf("k-means finished iterations: %d\n", gpu_result.iterations);
    
#ifdef DEBUG
    // Print centroids
    double* centroids = gpu_result.centroids;
    for (int i = 0; i < k; i++) {
        printf("Centroid %d: ", i);
        for (int j = 0; j < d; j++) {
            printf("%f ", centroids[i * d + j]);
        }
        printf("\n");
    }
    // Print cluster assignments
    printf("Cluster assignments:\n");
    for (int i = 0; i < n; i++) {
        printf("Point %d assigned to cluster %d\n", i, clusters[i]);
    }
#endif

    // Free allocated memory
    optima_free_data(data, gpu_result.centroids, clusters);

    return 0;
}