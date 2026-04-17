// example main.c
#include <stdio.h>
#include <time.h>
#include "OptimaKmeans/optima_kmeans.h"

#define CSV_PATH "../data/final_processed.csv"
#define CHUNK_SIZE 100000

int main() {
    int k = 5;
    int max_iter = 10000;

    // Load data once (shared between both paths)
    double* data;
    int n, d;
    if (optima_load_data_csv(CSV_PATH, &data, &n, &d) != 0) {
        fprintf(stderr, "Failed to load data from CSV file\n");
        return 1;
    }

    struct timespec t0, t1;

    // ----- Streaming GPU path (with GPU memory auto-adaptation) -----
    int* streaming_clusters;
    optima_malloc_clusters(&streaming_clusters, n);

    clock_gettime(CLOCK_MONOTONIC, &t0);
    KMeansResult streaming_result = optima_kmeans_gpu_streaming(data, n, d, k, max_iter, streaming_clusters);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double streaming_elapsed_sec = (double)(t1.tv_sec - t0.tv_sec) +
                                   (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;
    printf("[streaming] (GPU auto-adapt) elapsed: %.6f s (%.3f ms), iterations: %d\n",
           streaming_elapsed_sec, streaming_elapsed_sec * 1e3, streaming_result.iterations);

    // ----- Legacy GPU path -----
    int* legacy_clusters;
    optima_malloc_clusters(&legacy_clusters, n);

    clock_gettime(CLOCK_MONOTONIC, &t0);
    KMeansResult legacy_result = optima_kmeans_gpu(data, n, d, k, max_iter, legacy_clusters);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double legacy_elapsed_sec = (double)(t1.tv_sec - t0.tv_sec) +
                                (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;
    printf("[legacy]    (GPU standard)    elapsed: %.6f s (%.3f ms), iterations: %d\n",
           legacy_elapsed_sec, legacy_elapsed_sec * 1e3, legacy_result.iterations);

    // Compute inertia for legacy path
    double legacy_inertia = 0.0;
    for (int i = 0; i < n; i++) {
        int cid = legacy_clusters[i];
        if (cid < 0) continue;
        for (int j = 0; j < d; j++) {
            double diff = data[i * d + j] - legacy_result.centroids[cid * d + j];
            legacy_inertia += diff * diff;
        }
    }

    // Compute inertia for streaming path
    double streaming_inertia = 0.0;
    for (int i = 0; i < n; i++) {
        int cid = streaming_clusters[i];
        if (cid < 0) continue;
        for (int j = 0; j < d; j++) {
            double diff = data[i * d + j] - streaming_result.centroids[cid * d + j];
            streaming_inertia += diff * diff;
        }
    }

    printf("[legacy]    inertia: %.10f\n", legacy_inertia);
    printf("[streaming] inertia: %.10f\n", streaming_inertia);

#ifdef DEBUG
    double* centroids = gpu_result.centroids;
    for (int i = 0; i < k; i++) {
        printf("Centroid %d: ", i);
        for (int j = 0; j < d; j++) {
            printf("%f ", centroids[i * d + j]);
        }
        printf("\n");
    }
    printf("Cluster assignments:\n");
    for (int i = 0; i < n; i++) {
        printf("Point %d assigned to cluster %d\n", i, clusters[i]);
    }
#endif

    // Free allocated memory
    free(streaming_result.centroids);
    free(streaming_clusters);
    free(legacy_result.centroids);
    free(legacy_clusters);
    optima_free_data(data, NULL, NULL);

    return 0;
}