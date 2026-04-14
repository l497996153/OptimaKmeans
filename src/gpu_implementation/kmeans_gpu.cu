#include "kmeans_gpu.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

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

    for (int i = tid; i < K * D; i += blockDim.x) {
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
        if (lane == leader) atomicAdd(&d_new_centroids[cid * D + d], v);
    }
    if (lane == leader) atomicAdd(&d_counts[cid], __popc(group));
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

double* kmeans_gpu(double *h_data, int num_points, int dim, int k, int max_iteration, int *h_clusters, int *finished_iterations)
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

    // Allocate host buffer to compare centroids for convergence check
    double *host_buffer_prev_centroids = (double *)malloc((size_t)k * (size_t)dim * sizeof(double));
    memcpy(host_buffer_prev_centroids, h_initial_centroids, (size_t)k * (size_t)dim * sizeof(double));

    int iteration;
    for (iteration = 0; iteration < max_iteration; iteration++) {
        find_centroid<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(d_data, d_centroids, d_clusters, num_points, dim, k);
        cudaDeviceSynchronize();
        cudaMemset(d_new_centroids, 0, k * dim * sizeof(double));
        cudaMemset(d_counts, 0, k * sizeof(int));
        centroid_sum<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_clusters, d_new_centroids, d_counts, num_points, dim, k);
        cudaDeviceSynchronize();
        calculate_centroid<<<k, dim>>>(d_new_centroids, d_counts, dim, k);
        cudaDeviceSynchronize();
        cudaMemcpy(d_centroids, d_new_centroids, k * dim * sizeof(double), cudaMemcpyDeviceToDevice);

        // Check convergence by comparing new centroids to previous iteration
        double *host_buffer_new_centroids = (double *)malloc((size_t)k * (size_t)dim * sizeof(double));
        cudaMemcpy(host_buffer_new_centroids, d_centroids, k * dim * sizeof(double), cudaMemcpyDeviceToHost);
        int converged = 1;
        for (int i = 0; i < k * dim; i++) {
            if (fabs(host_buffer_new_centroids[i] - host_buffer_prev_centroids[i]) > 1e-6) {
                converged = 0;
                break;
            }
        }
        memcpy(host_buffer_prev_centroids, host_buffer_new_centroids, (size_t)k * (size_t)dim * sizeof(double));
        free(host_buffer_new_centroids);
        if (converged) 
        { 
            iteration++; 
            break; 
        }
    }

    free(host_buffer_prev_centroids);

    if (finished_iterations != NULL) {
        *finished_iterations = iteration;
    }

    double *h_centroids = (double *)malloc((size_t)k * (size_t)dim * sizeof(double));
    if (h_centroids != NULL) {
        cudaMemcpy(h_centroids, d_centroids, k * dim * sizeof(double), cudaMemcpyDeviceToHost);
    }
    cudaMemcpy(h_clusters, d_clusters, num_points * sizeof(int), cudaMemcpyDeviceToHost);

    free(h_initial_centroids);
    cudaFree(d_counts);
    cudaFree(d_new_centroids);
    cudaFree(d_centroids);
    cudaFree(d_clusters);
    cudaFree(d_data);

    return h_centroids;
}