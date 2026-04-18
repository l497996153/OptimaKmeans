#include "kmeans_gpu.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

/**
 * @brief Optimized Assignment Kernel
 * Uses sub-warp tiling: multiple points are handled by a single warp if D is small.
 */
__global__ void find_centroid(
    double *d_data,
    double *d_centroids,
    int *d_clusters,
    int *d_changed,
    int N,
    int D,
    int K)
{
    extern __shared__ double shared_centroids[];

    int tid = threadIdx.x;
    int lane = tid & 31;

    // Load centroids into shared memory
    for (int i = tid; i < K * D; i += blockDim.x) {
        shared_centroids[i] = d_centroids[i];
    }
    __syncthreads();

    // Determine sub-warp grouping logic
    int points_per_warp = (D <= 8) ? 4 : ((D <= 16) ? 2 : 1);
    int group_size = 32 / points_per_warp;

    int group_id_in_warp = lane / group_size;
    int lane_in_group = lane % group_size;

    int global_thread = blockIdx.x * blockDim.x + tid;
    int global_warp = global_thread >> 5;
    int point_idx = global_warp * points_per_warp + group_id_in_warp;

    if (point_idx >= N) return;

    // Create mask for shuffle operations within the sub-warp group
    unsigned group_mask;
    if (group_size == 32) {
        group_mask = 0xffffffffu;
    } else {
        group_mask = ((1u << group_size) - 1u) << (group_id_in_warp * group_size);
    }

    double best_dist = DBL_MAX;
    int best_k = 0;

    for (int k = 0; k < K; k++) {
        double partial_sum = 0.0;

        if (group_size < 32) {
            // Small Dim: Each thread in group handles one dimension
            if (lane_in_group < D) {
                double point_val = d_data[point_idx + lane_in_group * N];
                double centroid_val = shared_centroids[k * D + lane_in_group];
                double diff = point_val - centroid_val;
                partial_sum = diff * diff;
            }
        } else {
            // Large Dim: Full warp handles one point, looping over dimensions
            for (int d = lane; d < D; d += 32) {
                double point_val = d_data[point_idx + d * N];
                double centroid_val = shared_centroids[k * D + d];
                double diff = point_val - centroid_val;
                partial_sum += diff * diff;
            }
        }

        // Parallel reduction within the sub-warp group
        for (int offset = group_size >> 1; offset > 0; offset >>= 1) {
            partial_sum += __shfl_down_sync(group_mask, partial_sum, offset);
        }

        if (lane_in_group == 0) {
            if (partial_sum < best_dist) {
                best_dist = partial_sum;
                best_k = k;
            }
        }
    }

    // Leader of the sub-warp group updates assignment
    if (lane_in_group == 0) {
        if (d_clusters[point_idx] != best_k) {
            d_clusters[point_idx] = best_k;
            *d_changed = 1;
        }
    }
}

__global__ void centroid_sum(
    double *d_data,
    int *d_clusters,
    double *d_new_sums,
    int *d_counts,
    int N,
    int D,
    int K)
{
    extern __shared__ unsigned char shared_buffer[];

    int *shared_counts = (int *)shared_buffer;
    double *shared_sums = (double *)(shared_counts + K);

    int tid  = threadIdx.x;
    int idx  = blockIdx.x * blockDim.x + tid;
    int lane = tid & 31;

    for (int i = tid; i < K; i += blockDim.x) {
        shared_counts[i] = 0;
    }

    for (int i = tid; i < K * D; i += blockDim.x) {
        shared_sums[i] = 0.0;
    }
    __syncthreads();

    if (idx < N) {
        int cid = d_clusters[idx];
        if (cid >= 0 && cid < K) {
            unsigned active = __activemask();
            unsigned group  = __match_any_sync(active, cid);
            int leader = __ffs(group) - 1;

            if (lane == leader) {
                atomicAdd(&shared_counts[cid], __popc(group));
            }

            for (int d = 0; d < D; d++) {
                double my_val = d_data[d * N + idx];
                double sum = 0.0;

                unsigned temp = group;
                while (temp) {
                    int src_lane = __ffs(temp) - 1;
                    sum += __shfl_sync(active, my_val, src_lane);
                    temp &= (temp - 1);
                }

                if (lane == leader) {
                    atomicAdd(&shared_sums[cid * D + d], sum);
                }
            }
        }
    }
    __syncthreads();

    for (int i = tid; i < K; i += blockDim.x) {
        if (shared_counts[i] > 0) {
            atomicAdd(&d_counts[i], shared_counts[i]);
        }
    }

    for (int i = tid; i < K * D; i += blockDim.x) {
        if (shared_sums[i] != 0.0) {
            atomicAdd(&d_new_sums[i], shared_sums[i]);
        }
    }
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

double* kmeans_gpu(double *h_data, int num_points, int dim, int k, int max_iteration, int *h_clusters, int *finished_iterations, int threads_per_block)
{
    // 1. Setup Host memory
    double *host_initial_centroids = (double *)malloc((size_t)k * dim * sizeof(double));
    double *h_data_soa = (double *)malloc((size_t)num_points * dim * sizeof(double));

    for (int i = 0; i < num_points; i++) {
        for (int d = 0; d < dim; d++) {
            if (i < k) host_initial_centroids[i * dim + d] = h_data[i * dim + d];
            h_data_soa[d * num_points + i] = h_data[i * dim + d];
        }
    }
    memset(h_clusters, -1, num_points * sizeof(int));

    // 2. Device Allocation
    double *device_data, *device_centroids, *device_new_sums;
    int *device_clusters, *device_counts, *device_changed;
    
    cudaMalloc(&device_data, num_points * dim * sizeof(double));
    cudaMalloc(&device_centroids, k * dim * sizeof(double));
    cudaMalloc(&device_new_sums, k * dim * sizeof(double));
    cudaMalloc(&device_clusters, num_points * sizeof(int));
    cudaMalloc(&device_counts, k * sizeof(int));
    cudaMalloc(&device_changed, sizeof(int));

    cudaMemcpy(device_data, h_data_soa, num_points * dim * sizeof(double), cudaMemcpyHostToDevice);
    free(h_data_soa);

    cudaMemcpy(device_centroids, host_initial_centroids, k * dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_clusters, h_clusters, num_points * sizeof(int), cudaMemcpyHostToDevice);

    // 3. Configure Grid and Block dimensions
    int threadsPerBlock = 256;
    int warpsPerBlock = threadsPerBlock / 32;

    int pointsPerWarp = (dim <= 8) ? 4 : ((dim <= 16) ? 2 : 1);
    int pointsPerBlock = warpsPerBlock * pointsPerWarp;

    int blocksPerGrid_Find = (num_points + pointsPerBlock - 1) / pointsPerBlock;
    int blocksPerGrid_Sum  = (num_points + threadsPerBlock - 1) / threadsPerBlock;

    size_t shared_mem_find = k * dim * sizeof(double);
    size_t shared_mem_sum  = k * sizeof(int) + k * dim * sizeof(double);
    int iter;
    int h_changed;

    // 4. Main K-Means Loop
    for (iter = 0; iter < max_iteration; iter++) {
        h_changed = 0;
        cudaMemcpy(device_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice);

        // Assignment Phase
        find_centroid<<<blocksPerGrid_Find, threadsPerBlock, shared_mem_find>>>(device_data, device_centroids, device_clusters, device_changed, num_points, dim, k);

        cudaMemcpy(&h_changed, device_changed, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_changed == 0) break;

        // Reset for Update Phase
        cudaMemset(device_counts, 0, k * sizeof(int));
        cudaMemset(device_new_sums, 0, k * dim * sizeof(double));

        // Update Phase (Summation)
        centroid_sum<<<blocksPerGrid_Sum, threadsPerBlock, shared_mem_sum>>>(device_data, device_clusters, device_new_sums, device_counts, num_points, dim, k);
        
        // Update Phase (Division)
        calculate_centroid<<<k, dim>>>(device_new_sums, device_counts, dim, k);

        // Swap centroids for next iteration
        cudaMemcpy(device_centroids, device_new_sums, k * dim * sizeof(double), cudaMemcpyDeviceToDevice);
    }

    if (finished_iterations != NULL) *finished_iterations = iter;

    // 5. Final Retrieval
    double *h_final_centroids = (double *)malloc(k * dim * sizeof(double));
    cudaMemcpy(h_final_centroids, device_centroids, k * dim * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_clusters, device_clusters, num_points * sizeof(int), cudaMemcpyDeviceToHost);

    // 6. Cleanup
    free(host_initial_centroids);
    cudaFree(device_data);
    cudaFree(device_centroids);
    cudaFree(device_new_sums);
    cudaFree(device_clusters);
    cudaFree(device_counts);
    cudaFree(device_changed);

    return h_final_centroids;
}