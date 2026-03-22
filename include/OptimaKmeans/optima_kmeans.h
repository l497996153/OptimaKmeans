// Header file for OptimaKmeans library
#ifndef OPTIMA_KMEANS_H
#define OPTIMA_KMEANS_H

int optima_load_data_bin(const char* filename, double** data, int* n, int* d);
int optima_load_data_csv(const char* filename, double** data, int* n, int* d);
double* optima_kmeans(double *points, int num_points, int dim, int k, int max_iter, int *clusters);
void optima_free_data(double* data, double* centroids, int* clusters);
void optima_malloc_clusters(int** clusters, int k);

#endif // OPTIMA_KMEANS_H