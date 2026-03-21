// example main.c
#include <stdio.h>
#include "OptimaKmeans/optima_kmeans.h"

int main() {
    double* data;
    int n, d;

    // Load data from a csv file
    if (optima_load_data_csv("../data/X_scaled.csv", &data, &n, &d) != 0) {
        fprintf(stderr, "Failed to load data from CSV file\n");
        return 1;
    }

    // Run k-means
    int k = 5; // number of clusters
    int max_iter = 1000;
    int* clusters;
    optima_malloc_clusters(&clusters, n);
    double** centroids = optima_kmeans(data, n, d, k, max_iter, clusters);
    
    // Print centroids
    for (int i = 0; i < k; i++) {
        printf("Centroid %d: ", i);
        for (int j = 0; j < d; j++) {
            printf("%f ", centroids[i][j]);
        }
        printf("\n");
    }

    // Print cluster assignments
    for (int i = 0; i < n; i++) {
        printf("Point %d assigned to cluster %d\n", i, clusters[i]);
    }

    // Free allocated memory
    optima_free_data(data, centroids, clusters, k);

    return 0;
}