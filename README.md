# OptimaKmeans
A high-performance K-Means library accelerated by OpenMP (CPU) and CUDA (GPU).

## Memory Management

- All memory allocated by the library for data, centroids, and cluster assignments must be explicitly freed by the user to prevent memory leaks.
- The library provides a helper function `optima_free_data` to safely clean up all resources.

**Example**
See the [main.c](main.c) example for a complete usage demonstration.

**How To Run Example**
mkdir build
cd build
cmake ..
cmake --build . --config Release
./example

## How to run benchmark
Benchmarks use the `autotune` target (GPU timing on a CSV dataset) and an optional Python driver that sweeps kernel variants and thread-block sizes.
mkdir build
cd build
cmake ..
cmake --build . --config Release
benchmark run:
python3 scripts/benchmark.py with options
- `--csvpath`  pass dataset file path
- `--variants` limit the experiment approach such as 'cpu, gpu`
- `--repeats`  runs per configuration (default `3`)