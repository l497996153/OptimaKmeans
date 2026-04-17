#include "kmeans_gpu.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

extern "C" {
#include "../dataloader.h"
}

#define NUM_BUFFERS 3

/**
 * @brief Assign points to nearest centroid and track if any assignments changed
 *
 * @param d_data         Stores the data by row-major with [N * D]
 * @param d_centroids    Stores the centroids by row-major with [K * D]
 * @param d_clusters     Stores the cluster ID for each point with [N]
 * @param d_changed      Stores a flag indicating if any assignments changed with [1]
 * @param N              The number of points
 * @param D              The number of dimensions
 * @param K              The number of clusters 
 */
__global__ void find_centroid(float *d_data, float *d_centroids, int *d_clusters, int *d_changed, int N, int D, int K)
{
    
    // The shared memory has the current centroids for this iteration, so all threads in the block can access
    extern __shared__ float shared_centroids[];
    // The thread ID
    int tid = threadIdx.x;
    // The global ID of the thread in the grid
    int idx = blockIdx.x * blockDim.x + tid;

    // Load current centroids into shared memory
    for (int i = tid; i < K * D; i += blockDim.x) {
        shared_centroids[i] = d_centroids[i];
    }
    __syncthreads();

    if (idx < N) {
        float min_distance = 1e18f;
        int closest_centroid = 0;

        for (int k = 0; k < K; k++) {
            // Calculate the Euclidean distance between two points
            float current_distance = 0.0f;
            for (int d = 0; d < D; d++) {
                // We use row-major indexing
                float diff = d_data[idx * D + d] - shared_centroids[k * D + d];
                current_distance += diff * diff;
            }
            current_distance = sqrtf(current_distance);
            if (current_distance < min_distance) {
                min_distance = current_distance;
                closest_centroid = k;
            }
        }

        // If the centroid of a point has changed changed, we mark it and update the cluster in the next step
        if (d_clusters[idx] != closest_centroid) {
            d_clusters[idx] = closest_centroid;
            *d_changed = 1; 
        }
    }
}

/**
 * @brief Sum coordinates and update the number of points in each cluster so that we can compute new centroids in the next step
 *
 * @param d_data         Stores the data by row-major with [N * D]
 * @param d_clusters     Stores the cluster ID for each point with [N]
 * @param d_new_sums     Stores the sum of coordinates for each cluster with [K * D]
 * @param d_counts       Stores the number of points in each cluster with [K]
 * @param N              The number of points
 * @param D              The number of dimensions
 */
__global__ void centroid_sum(float *d_data, int *d_clusters, float *d_new_sums, int *d_counts, int N, int D)
{
    // For each thread, it will process one data point and update the corresponding cluster's sum and count
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // The cluster ID of the current data point
        int cluster_id = d_clusters[idx];
        // If this cluster ID is valid
        if (cluster_id >= 0) {
            // Atomically add the point's coordinates to the cluster's sum and increment the count
            atomicAdd(&d_counts[cluster_id], 1);
            for (int d = 0; d < D; d++) {
                atomicAdd(&d_new_sums[cluster_id * D + d], d_data[idx * D + d]);
            }
        }
    }
}

/**
* @brief Update centroids by dividing the sum of coordinates by the count of points in each cluster
*
* @param d_centroids    Stores the centroids by row-major with [K * D]
* @param d_new_sums     Stores the sum of coordinates for each cluster with [K * D]
* @param d_counts       Stores the number of points in each cluster with [K]
* @param K              The number of clusters 
* @param D              The number of dimensions
*/
__global__ void update_centroids(float *d_centroids, float *d_new_sums, int *d_counts, int K, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread will update one coordinate of one centroid
    if (idx < K * D) {
        int cluster_id = idx / D;
        int count = d_counts[cluster_id];
        if (count > 0) {
            d_centroids[idx] = d_new_sums[idx] / (float)count;
        }
    }
}

float* kmeans_gpu_float(float *h_data, int num_points, int dim, int k, int max_iteration, int *h_clusters, int *finished_iterations)
{
    float *host_initial_centroids = (float *)malloc((size_t)k * dim * sizeof(float));
    for (int i = 0; i < k; i++) {
        for (int d = 0; d < dim; d++) {
            host_initial_centroids[i * dim + d] = h_data[i * dim + d];
        }
    }
    memset(h_clusters, -1, num_points * sizeof(int));

    float *device_data, *device_centroids, *device_new_sums;
    int *device_clusters, *device_counts, *device_changed;
    
    cudaMalloc(&device_data, num_points * dim * sizeof(float));
    cudaMalloc(&device_centroids, k * dim * sizeof(float));
    cudaMalloc(&device_new_sums, k * dim * sizeof(float));
    cudaMalloc(&device_clusters, num_points * sizeof(int));
    cudaMalloc(&device_counts, k * sizeof(int));
    cudaMalloc(&device_changed, sizeof(int));

    cudaMemcpy(device_data, h_data, num_points * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_centroids, host_initial_centroids, k * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_clusters, h_clusters, num_points * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid_Points = (num_points + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid_Centroids = (k * dim + threadsPerBlock - 1) / threadsPerBlock;
    size_t shared_mem_size = k * dim * sizeof(float);

    int iter;
    int h_changed;

    for (iter = 0; iter < max_iteration; iter++) {
        h_changed = 0;
        cudaMemcpy(device_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice);

        find_centroid<<<blocksPerGrid_Points, threadsPerBlock, shared_mem_size>>>(
            device_data, device_centroids, device_clusters, device_changed, num_points, dim, k);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_changed, device_changed, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_changed == 0) {
            break;
        }

        cudaMemset(device_counts, 0, k * sizeof(int));
        cudaMemset(device_new_sums, 0, k * dim * sizeof(float));

        centroid_sum<<<blocksPerGrid_Points, threadsPerBlock>>>(
            device_data, device_clusters, device_new_sums, device_counts, num_points, dim);
        cudaDeviceSynchronize();

        update_centroids<<<blocksPerGrid_Centroids, threadsPerBlock>>>(
            device_centroids, device_new_sums, device_counts, k, dim);
        cudaDeviceSynchronize();
    }

    if (finished_iterations != NULL) {
        *finished_iterations = iter;
    }

    float *h_final_centroids = (float *)malloc(k * dim * sizeof(float));
    cudaMemcpy(h_final_centroids, device_centroids, k * dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_clusters, device_clusters, num_points * sizeof(int), cudaMemcpyDeviceToHost);

    free(host_initial_centroids);
    cudaFree(device_data);
    cudaFree(device_centroids);
    cudaFree(device_new_sums);
    cudaFree(device_clusters);
    cudaFree(device_counts);
    cudaFree(device_changed);

    return h_final_centroids;
}

// ---------------------------------------------------------------------------
// Streaming GPU K-Means: with GPU memory auto-adaptation.
// Fast path: all data fits on GPU.
// Fallback: chunked streaming pipeline when data exceeds GPU memory.
// ---------------------------------------------------------------------------
float* kmeans_gpu_streaming_float(float *h_data, int num_points, int dim, int k, int max_iteration,
                                  int *h_clusters, int *finished_iterations)
{
    size_t full_data_bytes_f = (size_t)num_points * dim * sizeof(float);

    // Initialize cluster assignments to -1 (unassigned)
    memset(h_clusters, -1, (size_t)num_points * sizeof(int));

    // --- Persistent device arrays ---
    float *d_centroids, *d_new_sums;
    int    *d_clusters_dev, *d_counts, *d_changed;

    cudaMalloc(&d_centroids,    (size_t)k * dim * sizeof(float));
    cudaMalloc(&d_new_sums,     (size_t)k * dim * sizeof(float));
    cudaMalloc(&d_clusters_dev, (size_t)num_points * sizeof(int));
    cudaMalloc(&d_counts,       (size_t)k * sizeof(int));
    cudaMalloc(&d_changed,      sizeof(int));

    cudaMemset(d_clusters_dev, -1, (size_t)num_points * sizeof(int));
    cudaMemcpy(d_centroids, h_data, (size_t)k * dim * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock         = THREADS_PER_BLOCK;
    int blocksPerGrid_Points    = (num_points + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid_Centroids = (k * dim + threadsPerBlock - 1) / threadsPerBlock;
    size_t shared_mem_size      = (size_t)k * dim * sizeof(float);

    int iter;
    int h_changed;

    // --- Try to allocate full dataset on GPU ---
    float *d_data_full = NULL;
    cudaError_t alloc_err = cudaMalloc(&d_data_full, full_data_bytes_f);
    int gpu_resident = (alloc_err == cudaSuccess);

    if (gpu_resident) {
        // ===== FAST PATH: data fits in GPU memory — copy once, iterate on GPU =====
        fprintf(stderr, "[streaming] GPU-resident fast path: %.1f MB on device\n",
                (double)full_data_bytes_f / (1024.0 * 1024.0));

        cudaMemcpy(d_data_full, h_data, full_data_bytes_f, cudaMemcpyHostToDevice);

        for (iter = 0; iter < max_iteration; iter++) {
            h_changed = 0;
            cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice);

            find_centroid<<<blocksPerGrid_Points, threadsPerBlock, shared_mem_size>>>(
                d_data_full, d_centroids, d_clusters_dev, d_changed, num_points, dim, k);
            cudaDeviceSynchronize();

            cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
            if (h_changed == 0) break;

            cudaMemset(d_counts,   0, (size_t)k * sizeof(int));
            cudaMemset(d_new_sums, 0, (size_t)k * dim * sizeof(float));

            centroid_sum<<<blocksPerGrid_Points, threadsPerBlock>>>(
                d_data_full, d_clusters_dev, d_new_sums, d_counts, num_points, dim);
            cudaDeviceSynchronize();

            update_centroids<<<blocksPerGrid_Centroids, threadsPerBlock>>>(
                d_centroids, d_new_sums, d_counts, k, dim);
            cudaDeviceSynchronize();
        }

        cudaFree(d_data_full);
    } else {
        // ===== FALLBACK: chunked streaming pipeline (data too large for GPU) =====
        cudaFree(d_data_full);
        d_data_full = NULL;

        int chunk_size = 100000;  // Default chunk size for fallback
        int num_chunks = (num_points + chunk_size - 1) / chunk_size;
        size_t chunk_data_bytes = (size_t)chunk_size * dim * sizeof(float);

        fprintf(stderr, "[streaming] Chunked fallback: %d chunks of %d rows\n",
                num_chunks, chunk_size);

        float *d_buffers[NUM_BUFFERS];
        cudaStream_t streams[NUM_BUFFERS];
        for (int i = 0; i < NUM_BUFFERS; i++) {
            cudaMalloc(&d_buffers[i], chunk_data_bytes);
            cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
        }

        int deviceToken[NUM_BUFFERS];

        #pragma omp parallel
        {
            #pragma omp single
            {
                for (iter = 0; iter < max_iteration; iter++) {
                    h_changed = 0;
                    cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice);
                    cudaMemset(d_counts,   0, (size_t)k * sizeof(int));
                    cudaMemset(d_new_sums, 0, (size_t)k * dim * sizeof(float));
                    memset(deviceToken, 0, sizeof(deviceToken));

                    for (int c = 0; c < num_chunks; c++) {
                        int slot     = c % NUM_BUFFERS;
                        int p_offset = c * chunk_size;
                        int cn       = (c < num_chunks - 1) ? chunk_size
                                                            : (num_points - p_offset);

                        #pragma omp task depend(inout: deviceToken[slot]) \
                                         firstprivate(slot, p_offset, cn)
                        {
                            size_t bytes = (size_t)cn * dim * sizeof(float);
                            int blks = (cn + threadsPerBlock - 1) / threadsPerBlock;

                            cudaMemcpyAsync(d_buffers[slot],
                                            h_data + (size_t)p_offset * dim,
                                            bytes, cudaMemcpyHostToDevice, streams[slot]);

                            find_centroid<<<blks, threadsPerBlock,
                                           shared_mem_size, streams[slot]>>>(
                                d_buffers[slot], d_centroids,
                                d_clusters_dev + p_offset,
                                d_changed, cn, dim, k);

                            centroid_sum<<<blks, threadsPerBlock, 0, streams[slot]>>>(
                                d_buffers[slot], d_clusters_dev + p_offset,
                                d_new_sums, d_counts, cn, dim);

                            cudaStreamSynchronize(streams[slot]);
                        }
                    }

                    #pragma omp taskwait
                    cudaDeviceSynchronize();

                    cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
                    if (h_changed == 0) break;

                    update_centroids<<<blocksPerGrid_Centroids, threadsPerBlock>>>(
                        d_centroids, d_new_sums, d_counts, k, dim);
                    cudaDeviceSynchronize();
                }
            }
        }

        for (int i = 0; i < NUM_BUFFERS; i++) {
            cudaFree(d_buffers[i]);
            cudaStreamDestroy(streams[i]);
        }
    }

    if (finished_iterations) *finished_iterations = iter;

    // --- Copy final results back ---
    float *h_final_centroids = (float *)malloc((size_t)k * dim * sizeof(float));
    cudaMemcpy(h_final_centroids, d_centroids, (size_t)k * dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_clusters, d_clusters_dev, (size_t)num_points * sizeof(int), cudaMemcpyDeviceToHost);

    // --- Cleanup ---
    cudaFree(d_centroids);
    cudaFree(d_new_sums);
    cudaFree(d_clusters_dev);
    cudaFree(d_counts);
    cudaFree(d_changed);

    return h_final_centroids;
}