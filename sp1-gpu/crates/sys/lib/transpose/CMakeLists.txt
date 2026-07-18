sp1_gpu_add_object_library(transpose_objs OBJECT
    transpose.cu
)
target_link_libraries(transpose_objs PRIVATE sp1_gpu_common)
