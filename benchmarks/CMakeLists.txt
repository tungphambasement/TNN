function(create_math_benchmark target_name source_file)
    add_executable(${target_name} ${source_file})

    # Link core TNN libraries
    set(BENCHMARK_LIBS
        tnn_nn
        tnn_math 
        tnn_tensor 
        tnn_matrix 
        tnn_utils
    )
    
    if(ENABLE_CUDA)
        list(APPEND BENCHMARK_LIBS tnn_cuda)
    endif()
    
    target_link_libraries(${target_name} PRIVATE ${BENCHMARK_LIBS})
endfunction()

create_default(serialization_benchmark serialization_benchmark.cpp)
create_default(compression_benchmark compression_benchmark.cpp)
create_math_benchmark(gemm_benchmark gemm_benchmark.cpp)
create_math_benchmark(tensor_ops_benchmark tensor_ops_benchmark.cpp)