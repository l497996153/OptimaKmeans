// Data Loader header file for k-means algorithm 
#ifndef DATALOADER_H
#define DATALOADER_H

#include <stdio.h>

// Load NxD data from a binary file or a CSV file
// Returns 0 on success, -1 on failure
int load_data_bin(const char* filename, double** data, int* n, int* d);
int load_data_csv(const char* filename, double** data, int* n, int* d);

// Free the allocated data
void free_data(double* data);

// --- Chunked CSV API for streaming pipelines ---

typedef struct {
    FILE* file;
    long  data_start_pos;
    int   n;  // total rows
    int   d;  // dimensions per row (columns minus first 2)
    int   rows_read; // running count of rows consumed so far
} CsvReader;

// Probe CSV metadata without loading data. Writes total rows -> *n, dims -> *d.
int csv_probe(const char* filename, int* n, int* d);

// Open a CSV file and position the cursor at the first data row.
// Caller must eventually call csv_reader_close().
int csv_reader_open(CsvReader* reader, const char* filename);

// Read up to max_rows into caller-supplied buffer (row-major, size >= max_rows*d doubles).
// Returns the number of rows actually read (0 at EOF, -1 on error).
int csv_read_chunk(CsvReader* reader, double* buf, int max_rows);

// Close the reader and release the FILE handle.
void csv_reader_close(CsvReader* reader);

// Rewind to the first data row so the file can be re-scanned for the next iteration.
void csv_reader_rewind(CsvReader* reader);

#endif // DATALOADER_H