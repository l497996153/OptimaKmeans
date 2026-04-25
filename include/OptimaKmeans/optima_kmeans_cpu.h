#ifndef OPTIMA_KMEANS_CPU_H
#define OPTIMA_KMEANS_CPU_H

int optima_load_data_bin(const char* filename, float** data, int* n, int* d);
int optima_load_data_csv(const char* filename, float** data, int* n, int* d);
float* optima_kmeans(float *points, int num_points, int dim, int k, int max_iter, int *clusters, int *iter_converge);
void optima_free_data(float* data, float* centroids, int* clusters);
void optima_malloc_clusters(int** clusters, int n);

#endif 
