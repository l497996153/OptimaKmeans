#ifndef KMEANS_GPU_H
#define KMEANS_GPU_H

#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

#ifdef __cplusplus
extern "C" {
#endif

// Assign each point to nearest centroid; sets *d_changed = 1 if any assignment changed.
__global__ void find_centroid(float *d_data, float *d_centroids, int *d_clusters,
                              int *d_changed, int N, int D, int K);

// Accumulate per-cluster coordinate sums and counts (via atomics).
__global__ void centroid_sum(float *d_data, int *d_clusters, float *d_new_sums,
                             int *d_counts, int N, int D);

// Divide accumulated sums by counts to produce new centroids.
__global__ void update_centroids(float *d_centroids, float *d_new_sums,
                                 int *d_counts, int K, int D);

// Host-side GPU entry point (whole-dataset in memory). Returns centroid array [K*D].
float* kmeans_gpu_float(float *h_data, int num_points, int dim, int k, int max_iteration,
                        int *h_clusters, int *finished_iterations);

// Streaming GPU entry point: with GPU memory auto-adaptation (fast path or chunked fallback).
// Takes pre-loaded float data. Returns centroid array [K*D].
float* kmeans_gpu_streaming_float(float *h_data, int num_points, int dim, int k, int max_iteration,
                                  int *h_clusters, int *finished_iterations);

#ifdef __cplusplus
}
#endif

#endif