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
    
    // Skip the header line
    if (!fgets(line, sizeof(line), file)) {
        fclose(file);
        return -1;
    }

    *n = 0;
    *d = 0;
    int first_data_line = 1;

    // Count the number of lines and columns in the file
    // and we do not read the first two columns
    long data_start_pos = ftell(file); 
    while (fgets(line, sizeof(line), file)) {
        (*n)++;
        if (first_data_line) {
            int col_count = 0;
            char* tmp = strdup(line);
            char* token = strtok(tmp, ",");
            while (token) {
                col_count++;
                token = strtok(NULL, ",");
            }
            free(tmp);
            *d = col_count - 2;
            first_data_line = 0;
        }
    }
    
    fseek(file, data_start_pos, SEEK_SET);

    // Allocate memory for data
    *data = (double*)malloc((size_t)(*n) * (size_t)(*d) * sizeof(double));
    if (!*data) {
        fclose(file);
        return -1;
    }

    // Read the data from the file
    int i = 0;
    while (fgets(line, sizeof(line), file) && i < *n) {
        char* token = strtok(line, ","); 
        
        if (token) {
            token = strtok(NULL, ","); 
        }
        
        int j = 0;
        while (j < *d) {
            token = strtok(NULL, ",\n\r");
            if (token) {
                (*data)[i * (*d) + j] = strtod(token, NULL);
            }
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

// --- Chunked CSV API implementation ---

int csv_probe(const char* filename, int* n, int* d) {
    FILE* file = fopen(filename, "r");
    if (!file) return -1;

    char line[MAX_LINE_LEN];
    if (!fgets(line, sizeof(line), file)) { fclose(file); return -1; }

    *n = 0;
    *d = 0;
    int first = 1;
    while (fgets(line, sizeof(line), file)) {
        (*n)++;
        if (first) {
            int col_count = 0;
            char* tmp = strdup(line);
            char* token = strtok(tmp, ",");
            while (token) { col_count++; token = strtok(NULL, ","); }
            free(tmp);
            *d = col_count - 2;
            first = 0;
        }
    }
    fclose(file);
    return 0;
}

int csv_reader_open(CsvReader* reader, const char* filename) {
    if (!reader) return -1;
    reader->file = fopen(filename, "r");
    if (!reader->file) return -1;

    char line[MAX_LINE_LEN];
    if (!fgets(line, sizeof(line), reader->file)) {
        fclose(reader->file);
        reader->file = NULL;
        return -1;
    }

    reader->n = 0;
    reader->d = 0;
    reader->rows_read = 0;
    int first = 1;
    long pos_after_header = ftell(reader->file);

    while (fgets(line, sizeof(line), reader->file)) {
        reader->n++;
        if (first) {
            int col_count = 0;
            char* tmp = strdup(line);
            char* token = strtok(tmp, ",");
            while (token) { col_count++; token = strtok(NULL, ","); }
            free(tmp);
            reader->d = col_count - 2;
            first = 0;
        }
    }

    reader->data_start_pos = pos_after_header;
    fseek(reader->file, reader->data_start_pos, SEEK_SET);
    return 0;
}

int csv_read_chunk(CsvReader* reader, double* buf, int max_rows) {
    if (!reader || !reader->file || !buf) return -1;

    char line[MAX_LINE_LEN];
    int rows = 0;
    int d = reader->d;

    while (rows < max_rows && fgets(line, sizeof(line), reader->file)) {
        char* token = strtok(line, ",");
        if (token) token = strtok(NULL, ","); // skip first two columns

        for (int j = 0; j < d; j++) {
            token = strtok(NULL, ",\n\r");
            if (token) {
                buf[rows * d + j] = strtod(token, NULL);
            }
        }
        rows++;
    }
    reader->rows_read += rows;
    return rows;
}

void csv_reader_close(CsvReader* reader) {
    if (reader && reader->file) {
        fclose(reader->file);
        reader->file = NULL;
    }
}

void csv_reader_rewind(CsvReader* reader) {
    if (reader && reader->file) {
        fseek(reader->file, reader->data_start_pos, SEEK_SET);
        reader->rows_read = 0;
    }
}