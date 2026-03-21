# OptimaKmeans
A high-performance K-Means library accelerated by OpenMP (CPU) and CUDA (GPU).

## Memory Management

- All memory allocated by the library for data, centroids, and cluster assignments must be explicitly freed by the user to prevent memory leaks.
- The library provides a helper function `optima_free_data` to safely clean up all resources.

**Example**
See the [main.c](main.c) example for a complete usage demonstration.