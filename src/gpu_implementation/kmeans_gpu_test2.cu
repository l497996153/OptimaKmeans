#include "kmeans_gpu.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__global__ void find_centroid(double *device_data, double *device_centroids, int *device_clusters, int *device_cluster_changed, int N, int dimensions, int K)
{
    extern __shared__ double shared_centroids[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    for (int i = tid; i < K * dimensions; i += blockDim.x) {
        shared_centroids[i] = device_centroids[i];
    }
    __syncthreads();

    if (idx < N) {
        double min_distance = 1e18;
        int closest_centroid = 0;

        for (int k = 0; k < K; k++) {
            double current_distance = 0.0;
            for (int d = 0; d < dimensions; ++d) {
                double diff = device_data[idx + (d * N)] - shared_centroids[k * dimensions + d];
                current_distance += diff * diff;
            }
            if (current_distance < min_distance) {
                min_distance = current_distance;
                closest_centroid = k;
            }
        }

        if (device_clusters[idx] != closest_centroid) {
            device_clusters[idx] = closest_centroid;
            *device_cluster_changed = 1;
        }
    }
}

__global__ void centroid_sum_partial(
    double *device_data,
    int *device_clusters,
    double *device_partial_sums,
    int *device_partial_counts,
    int N,
    int dimensions,
    int K)
{
    extern __shared__ unsigned char shared_buffer[];

    int *shared_counts = (int *)shared_buffer;
    size_t counts_bytes = (size_t)K * sizeof(int);
    size_t sums_offset = (counts_bytes + sizeof(double) - 1) & ~(sizeof(double) - 1);
    double *shared_sums = (double *)(shared_buffer + sums_offset);

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int bid = blockIdx.x;

    for (int i = tid; i < K; i += blockDim.x) {
        shared_counts[i] = 0;
    }

    for (int i = tid; i < K * dimensions; i += blockDim.x) {
        shared_sums[i] = 0.0;
    }

    __syncthreads();

    if (idx < N) {
        int cluster_id = device_clusters[idx];

        if (cluster_id >= 0 && cluster_id < K) {
            for (int d = 0; d < dimensions; d++) {
                double v = device_data[idx + d * N];
                atomicAdd(&shared_sums[cluster_id * dimensions + d], v);
            }
            atomicAdd(&shared_counts[cluster_id], 1);
        }
    }

    __syncthreads();

    for (int i = tid; i < K; i += blockDim.x) {
        device_partial_counts[bid * K + i] = shared_counts[i];
    }

    for (int i = tid; i < K * dimensions; i += blockDim.x) {
        device_partial_sums[(size_t)bid * K * dimensions + i] = shared_sums[i];
    }
}

__global__ void reduce_partials(
    double *device_partial_sums,
    int *device_partial_counts,
    double *device_new_sums,
    int *device_num_point_each_cluster,
    int num_blocks,
    int dimensions,
    int K)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < K * dimensions; i += stride) {
        double total = 0.0;
        for (int b = 0; b < num_blocks; b++) {
            total += device_partial_sums[(size_t)b * K * dimensions + i];
        }
        device_new_sums[i] = total;
    }

    for (int k = tid; k < K; k += stride) {
        int total = 0;
        for (int b = 0; b < num_blocks; b++) {
            total += device_partial_counts[b * K + k];
        }
        device_num_point_each_cluster[k] = total;
    }
}

__global__ void calculate_centroid(double *device_new_sums, int *device_num_point_each_cluster, int dimensions, int K)
{
    int cid = blockIdx.x;
    int d = threadIdx.x;

    if (cid >= K || d >= dimensions) {
        return;
    }

    int count = device_num_point_each_cluster[cid];
    if (count > 0) {
        device_new_sums[cid * dimensions + d] /= count;
    }
}

double* kmeans_gpu(double *h_data, int num_points, int dimension, int k, int max_iteration,
                   int *host_clusters, int *finished_iterations, int threads_per_block)
{
    double *host_initial_centroids = (double *)malloc((size_t)k * dimension * sizeof(double));
    double *host_data_column_major_order = (double *)malloc((size_t)num_points * dimension * sizeof(double));

    for (int i = 0; i < num_points; i++) {
        for (int d = 0; d < dimension; d++) {
            if (i < k) {
                host_initial_centroids[i * dimension + d] = h_data[i * dimension + d];
            }
            host_data_column_major_order[d * num_points + i] = h_data[i * dimension + d];
        }
    }

    memset(host_clusters, -1, num_points * sizeof(int));

    double *device_data;
    double *device_centroids;
    double *device_new_sums;
    double *device_partial_sums;
    int *device_clusters;
    int *device_num_point_each_cluster;
    int *device_partial_counts;
    int *device_cluster_changed;

    cudaMalloc(&device_data, (size_t)num_points * dimension * sizeof(double));
    cudaMalloc(&device_centroids, (size_t)k * dimension * sizeof(double));
    cudaMalloc(&device_new_sums, (size_t)k * dimension * sizeof(double));
    cudaMalloc(&device_clusters, (size_t)num_points * sizeof(int));
    cudaMalloc(&device_num_point_each_cluster, (size_t)k * sizeof(int));
    cudaMalloc(&device_cluster_changed, sizeof(int));

    cudaMemcpy(device_data, host_data_column_major_order, (size_t)num_points * dimension * sizeof(double), cudaMemcpyHostToDevice);
    free(host_data_column_major_order);

    cudaMemcpy(device_centroids, host_initial_centroids, (size_t)k * dimension * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_clusters, host_clusters, (size_t)num_points * sizeof(int), cudaMemcpyHostToDevice);

    int thread_per_block = (threads_per_block > 0) ? threads_per_block : 256;
    int blocksPerGrid_Points = (num_points + thread_per_block - 1) / thread_per_block;

    cudaMalloc(&device_partial_sums, (size_t)blocksPerGrid_Points * k * dimension * sizeof(double));
    cudaMalloc(&device_partial_counts, (size_t)blocksPerGrid_Points * k * sizeof(int));

    size_t find_shared_mem_size = (size_t)k * dimension * sizeof(double);

    size_t shared_counts_bytes = (size_t)k * sizeof(int);
    size_t shared_sums_offset = (shared_counts_bytes + sizeof(double) - 1) & ~(sizeof(double) - 1);
    size_t centroid_sum_shared_mem = shared_sums_offset + (size_t)k * dimension * sizeof(double);

    int iteration;
    int host_changed;

    for (iteration = 0; iteration < max_iteration; iteration++) {
        host_changed = 0;
        cudaMemcpy(device_cluster_changed, &host_changed, sizeof(int), cudaMemcpyHostToDevice);

        find_centroid<<<blocksPerGrid_Points, thread_per_block, find_shared_mem_size>>>(
            device_data,
            device_centroids,
            device_clusters,
            device_cluster_changed,
            num_points,
            dimension,
            k);

        cudaDeviceSynchronize();

        cudaMemcpy(&host_changed, device_cluster_changed, sizeof(int), cudaMemcpyDeviceToHost);

        if (host_changed == 0) {
            break;
        }

        centroid_sum_partial<<<blocksPerGrid_Points, thread_per_block, centroid_sum_shared_mem>>>(
            device_data,
            device_clusters,
            device_partial_sums,
            device_partial_counts,
            num_points,
            dimension,
            k);

        cudaDeviceSynchronize();

        int reduce_threads = 256;
        int reduce_blocks = 256;

        reduce_partials<<<reduce_blocks, reduce_threads>>>(
            device_partial_sums,
            device_partial_counts,
            device_new_sums,
            device_num_point_each_cluster,
            blocksPerGrid_Points,
            dimension,
            k);

        cudaDeviceSynchronize();

        calculate_centroid<<<k, dimension>>>(
            device_new_sums,
            device_num_point_each_cluster,
            dimension,
            k);

        cudaDeviceSynchronize();

        cudaMemcpy(device_centroids, device_new_sums, (size_t)k * dimension * sizeof(double), cudaMemcpyDeviceToDevice);
    }

    if (finished_iterations != NULL) {
        *finished_iterations = iteration;
    }

    double *host_final_centroids = (double *)malloc((size_t)k * dimension * sizeof(double));

    cudaMemcpy(host_final_centroids, device_centroids, (size_t)k * dimension * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_clusters, device_clusters, (size_t)num_points * sizeof(int), cudaMemcpyDeviceToHost);

    free(host_initial_centroids);

    cudaFree(device_data);
    cudaFree(device_centroids);
    cudaFree(device_new_sums);
    cudaFree(device_partial_sums);
    cudaFree(device_clusters);
    cudaFree(device_num_point_each_cluster);
    cudaFree(device_partial_counts);
    cudaFree(device_cluster_changed);

    return host_final_centroids;
}