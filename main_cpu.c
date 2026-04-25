// example main.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "OptimaKmeans/optima_kmeans_cpu.h"

int main(int argc, char* argv[]) {

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <percentage>\n", argv[0]);
        return -1;
    }

    double percentage = atof(argv[1]);

    float* data;
    int n, d;
    struct timespec start, end;

    // Load data from a csv file
    // Skip 1 header row + 2 leading columns (Driver, LapNumber) to match kmeans_base.py
    if (optima_load_data_csv("/afs/ece.cmu.edu/usr/zhuoqili/Private/OptimaKmeans/dataset/data/f1_data/processed/final_processed.csv", &data, &n, &d) != 0) {
        fprintf(stderr, "Failed to load data from CSV file\n");
        return 1;
    }

    // Use only a percentage of the data
    n = (int)(n * percentage);

    // Run k-means
    int k = 5; // number of clusters
    int max_iter = 500;
    int* clusters;
    int iter_converge = max_iter;
    optima_malloc_clusters(&clusters, n);
    clock_gettime(CLOCK_MONOTONIC, &start);
    float* centroids = optima_kmeans(data, n, d, k, max_iter, clusters, &iter_converge);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_ms = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1e6;

    double inertia = 0.0;
    for (int i = 0; i < n; i++) {
        int c = clusters[i];
        double sum_sq = 0.0;
        for (int j = 0; j < d; j++) {
            double diff = data[i * d + j] - centroids[c * d + j];
            sum_sq += diff * diff;
        }
        inertia += sum_sq;
    }

    //iter_converge = iter_converge + 1;
    printf("Time: %.2f ms, Iterations: %d, Time per Iteration: %.2f ms, Inertia: %.6f\n", time_ms, iter_converge, iter_converge > 0 ? time_ms / iter_converge : 0.0, inertia);

    // Write centroids and cluster assignments to CSV for Python plotting.
    FILE* fc = fopen("/afs/ece.cmu.edu/usr/zhuoqili/Private/OptimaKmeans/centroids.csv", "w");
    if (fc) {
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < d; j++) {
                fprintf(fc, "%f%s", centroids[i * d + j], j == d - 1 ? "\n" : ",");
            }
        }
        fclose(fc);
    }
    FILE* fa = fopen("/afs/ece.cmu.edu/usr/zhuoqili/Private/OptimaKmeans/clusters.csv", "w");
    if (fa) {
        for (int i = 0; i < n; i++) {
            fprintf(fa, "%d\n", clusters[i]);
        }
        fclose(fa);
    }



#ifdef DEBUG
    // Print centroids
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
    optima_free_data(data, centroids, clusters);
    return 0;
}
