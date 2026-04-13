#include "kmeans_gpu.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <omp.h>

// Finding the nearest centroid for every point
__global__ void find_centroid(double *d_data, double *d_centroids, int *d_clusters, int N, int D, int K)
{
    //TODO
    // Load current centroids in shared memory
    extern __shared__ double shared_centroids[];
    // The thread id in the thread block
    int tid = threadIdx.x;
    // The thread id in the grid
    int idx = blockIdx.x * blockDim.x + tid;

    for (int i = tid; i < K *D; i += blockDim.x) {
        shared_centroids[i] = d_centroids[i];
    }

    // Make sure all threads load centroids in shared memory
    __syncthreads();

    if (idx < N) {
        double smallest_distance = DBL_MAX;
        int closest_centroid = -1;

        for (int k = 0; k < K; k++) {
            double current_distance = 0.0;
            for (int d = 0; d < D; d++) {
                // Calculates the Euclidean distance between two points
                double point_val = d_data[idx + (d * N)];
                double centroid_val = shared_centroids[k * D + d];
                current_distance += (point_val - centroid_val) * (point_val - centroid_val);
            }
            if (current_distance < smallest_distance) {
                smallest_distance = current_distance;
                closest_centroid = k;
            }
        }
        d_clusters[idx] = closest_centroid;
    }
}

// Calculating the centroids
__global__ void centroid_sum(double *d_data, int *d_clusters, double *d_new_centroids, int *d_counts, int N, int D, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int cid = d_clusters[idx];
    if (cid < 0 || cid >= K) return;
    unsigned int active = __activemask();
    unsigned group = __match_any_sync(active, cid);
    int lane = threadIdx.x & 31;
    int leader = __ffs(group) - 1;
    for (int d = 0; d < D; ++d) {
        double v = d_data[idx + (d * N)];
        for (int offset = 16; offset > 0; offset >>= 1) {
            double other = __shfl_down_sync(active, v, offset);
            int partner = lane + offset;
            if (partner < 32 && ((group >> partner) & 1))
                v += other;
        }
        if (lane==leader) atomicAdd(&d_new_centroids[cid * D + d], v);
    }
    if (lane==leader) atomicAdd(&d_counts[cid], __popc(group));
}

__global__ void calculate_centroid(double *d_new_centroids, int *d_counts, int D, int K)
{
    int cid = blockIdx.x;
    int d = threadIdx.x;
    if (cid >= K || d >= D) return;
    int count = d_counts[cid];
    if (count > 0) {
        d_new_centroids[cid * D + d] /= count;
    }
    
}

double* kmeans_gpu(double *h_data, int num_points, int dim, int k, int max_iteration, int *h_clusters)
{
    double *h_initial_centroids = (double *)malloc((size_t)k * (size_t)dim * sizeof(double));
    if (h_initial_centroids == NULL) {
        return NULL;
    }

    for (int cid = 0; cid < k; cid++) {
        int src_idx = cid % num_points;
        for (int d = 0; d < dim; d++) {
            h_initial_centroids[cid * dim + d] = h_data[src_idx + (d * num_points)];
        }
    }

    double *d_data;
    int *d_clusters;
    double *d_new_centroids;
    double *d_centroids;
    int *d_counts;

    cudaMalloc(&d_data, num_points * dim * sizeof(double));
    cudaMalloc(&d_clusters, num_points * sizeof(int));
    cudaMalloc(&d_new_centroids, k * dim * sizeof(double));
    cudaMalloc(&d_centroids, k * dim * sizeof(double));
    cudaMalloc(&d_counts, k * sizeof(int));

    cudaMemcpy(d_data, h_data, num_points * dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_initial_centroids, k * dim * sizeof(double), cudaMemcpyHostToDevice);

    size_t shared_mem_size = k * dim * sizeof(double);
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_points + threadsPerBlock - 1) / threadsPerBlock;

    cudaStream_t stream_find, stream_accum;
    cudaStreamCreateWithFlags(&stream_find, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream_accum, cudaStreamNonBlocking);

    cudaEvent_t *evt_find = (cudaEvent_t *)malloc(max_iteration * sizeof(cudaEvent_t));
    cudaEvent_t *evt_norm = (cudaEvent_t *)malloc(max_iteration * sizeof(cudaEvent_t));
    for (int i = 0; i < max_iteration; i++) {
        cudaEventCreate(&evt_find[i]);
        cudaEventCreate(&evt_norm[i]);
    }
    // OMP dependency tokens
    int *token_find  = (int *)malloc(max_iteration * sizeof(int));
    int *token_reset = (int *)malloc(max_iteration * sizeof(int));
    int *token_accum = (int *)malloc(max_iteration * sizeof(int));
    int ready = 1;

    #pragma omp parallel
    #pragma omp single
    {
        int *prev_accum = &ready;

        for (int i = 0; i < max_iteration; i++) {
            // --- TASK FIND(i) ---
            #pragma omp task firstprivate(i, prev_accum) \
                depend(in: *prev_accum) depend(out: token_find[i])
            {
                if (i > 0) cudaStreamWaitEvent(stream_find, evt_norm[i - 1], 0);
                find_centroid<<<blocksPerGrid, threadsPerBlock, shared_mem_size, stream_find>>>(
                    d_data, d_centroids, d_clusters, num_points, dim, k);
                cudaEventRecord(evt_find[i], stream_find);
            }

            // --- TASK RESET(i) ---
            #pragma omp task firstprivate(i, prev_accum) \
                depend(in: *prev_accum) depend(out: token_reset[i])
            {
                cudaMemsetAsync(d_new_centroids, 0, (size_t)k * dim * sizeof(double), stream_accum);
                cudaMemsetAsync(d_counts,        0, (size_t)k        * sizeof(int),    stream_accum);
            }

            // --- TASK ACCUM(i) ---
            #pragma omp task firstprivate(i) \
                depend(in: token_find[i], token_reset[i]) depend(out: token_accum[i])
            {
                cudaStreamWaitEvent(stream_accum, evt_find[i], 0);
                centroid_sum<<<blocksPerGrid, threadsPerBlock, 0, stream_accum>>>(
                    d_data, d_clusters, d_new_centroids, d_counts, num_points, dim, k);
                calculate_centroid<<<k, dim, 0, stream_accum>>>(d_new_centroids, d_counts, dim, k);
                cudaMemcpyAsync(d_centroids, d_new_centroids,
                                (size_t)k * dim * sizeof(double), cudaMemcpyDeviceToDevice, stream_accum);
                cudaEventRecord(evt_norm[i], stream_accum);
            }

            prev_accum = &token_accum[i];
        }
    }

    cudaDeviceSynchronize();

    double *h_centroids = (double *)malloc((size_t)k * (size_t)dim * sizeof(double));
    if (h_centroids != NULL) {
        cudaMemcpy(h_centroids, d_centroids, k * dim * sizeof(double), cudaMemcpyDeviceToHost);
    }
    cudaMemcpy(h_clusters, d_clusters, num_points * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < max_iteration; i++) {
        cudaEventDestroy(evt_find[i]);
        cudaEventDestroy(evt_norm[i]);
    }
    cudaStreamDestroy(stream_find);
    cudaStreamDestroy(stream_accum);

    free(token_find);
    free(token_reset);
    free(token_accum);
    free(h_initial_centroids);
    cudaFree(d_counts);
    cudaFree(d_new_centroids);
    cudaFree(d_centroids);
    cudaFree(d_clusters);
    cudaFree(d_data);

    return h_centroids;
}