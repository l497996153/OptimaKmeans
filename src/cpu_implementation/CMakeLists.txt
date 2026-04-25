find_package(OpenMP REQUIRED)

if(C_version STREQUAL "1")
    set(KMEANS_SRC kmeans_v1.c)
elseif(C_version STREQUAL "2")
    set(KMEANS_SRC kmeans_v2.c)
elseif(C_version STREQUAL "3")
    set(KMEANS_SRC kmeans_v3.c)
endif()

add_library(optima_kmeans optima_kmeans.c dataloader.c ${KMEANS_SRC})
target_link_libraries(optima_kmeans PUBLIC OpenMP::OpenMP_C)
target_include_directories(optima_kmeans PUBLIC ${CMAKE_SOURCE_DIR}/include)
