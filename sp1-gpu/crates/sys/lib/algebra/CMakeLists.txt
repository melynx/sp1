sp1_gpu_add_object_library(algebra_objs OBJECT
    algebra.cu
)
target_link_libraries(algebra_objs PRIVATE sp1_gpu_common)
