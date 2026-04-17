cmake_minimum_required(VERSION 3.10)
project(OptimaKmeans LANGUAGES C CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

add_subdirectory(src)
add_executable(example main.c)
target_link_libraries(example optima_kmeans m OpenMP::OpenMP_C OpenMP::OpenMP_CXX)