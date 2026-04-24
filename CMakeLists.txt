cmake_minimum_required(VERSION 3.10)

if(NOT DEFINED BUILD_TARGET)
    set(BUILD_TARGET "cpu" CACHE STRING "Build target: cpu or gpu")
endif()

if(BUILD_TARGET STREQUAL "gpu")
    project(OptimaKmeans LANGUAGES C CXX CUDA)
    set(CMAKE_CUDA_ARCHITECTURES native)
    find_package(CUDAToolkit REQUIRED)
else()
    project(OptimaKmeans LANGUAGES C)
endif()

add_subdirectory(src)

if(BUILD_TARGET STREQUAL "gpu")
    add_executable(example main_gpu.c)
    target_link_libraries(example optima_kmeans m)

    add_executable(autotune autotune.c)
    target_link_libraries(autotune optima_kmeans m)
else()
    add_executable(example main_cpu.c)
    target_link_libraries(example optima_kmeans m)
endif()
