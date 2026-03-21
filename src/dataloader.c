// Data loader implementation for k-means algorithm
#include "dataloader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_LEN 65536

// Load NxD data from a binary file into data
// Returns 0 on success, -1 on failure
int load_data_bin(const char* filename, double** data, int* n, int* d) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        return -1; // Failed to open file
    }
    // Load n and d from the file
    if (fread(n, sizeof(int), 1, file) != 1 || fread(d, sizeof(int), 1, file) != 1) {
        fclose(file);
        return -1;
    }
    // Allocate memory for data
    *data = (double*)malloc((*n) * (*d) * sizeof(double));
    if (!*data) {
        fclose(file);
        return -1;
    }
    // Read the data from the file
    if (fread(*data, sizeof(double), (*n) * (*d), file) != (*n) * (*d)) {
        free(*data);
        fclose(file);
        return -1;
    }

    fclose(file);
    return 0;
}


// Load NxD data from a CSV file into data
// Returns 0 on success, -1 on failure
int load_data_csv(const char* filename, double** data, int* n, int* d) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        return -1;
    }
    char line[MAX_LINE_LEN];
    *n = 0;
    *d = 0;

    int first_line = 1;
    int dim = 0;
    // Count n and d
    while (fgets(line, sizeof(line), file)) {
        (*n)++;
        int col_count = 0;
        char* tmp = strdup(line);
        char* token = strtok(tmp, ",");
        while (token) {
            col_count++;
            token = strtok(NULL, ",");
        }
        free(tmp);
        if (first_line) {
            dim = col_count;
            *d = col_count;
            first_line = 0;
        } else if (col_count != dim) {
            fclose(file);
            fprintf(stderr, "Inconsistent column count in CSV file\n");
            return -1;
        }
    }
    rewind(file);

    // Allocate memory for data
    *data = (double*)malloc((*n) * (*d) * sizeof(double));
    if (!*data) {
        fclose(file);
        return -1;
    }

    // Read the data from the file
    int i = 0;
    while (fgets(line, sizeof(line), file)) {
        char* token = strtok(line, ",");
        int j = 0;
        while (token && j < *d) {
            (*data)[i * (*d) + j] = strtod(token, NULL);
            token = strtok(NULL, ",");
            j++;
        }
        i++;
    }

    fclose(file);
    return 0;
}

// Free the allocated data
void free_data(double* data) {
    if (data) {
        free(data);
    }
}