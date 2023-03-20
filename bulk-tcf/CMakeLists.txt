cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(bulk_tcf VERSION 1.0 LANGUAGES CXX CUDA)

set(GPU_ARCHS "")
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/EvalGpuArchs.cmake)
  evaluate_gpu_archs(GPU_ARCHS)


find_package(CUDAToolkit 11.0 REQUIRED)

#macro for configuring executable
include(cmake/ConfigureExecutable.cmake)


set(DEFAULT_BUILD_TYPE "Release")




set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_COMPILER $ENV{CC})
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}") 


add_library(bulk_tcf INTERFACE)

target_include_directories(bulk_tcf INTERFACE ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(bulk_tcf INTERFACE ssl crypto)


ConfigureExecutable(tcf_wrapper_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/tcf_wrapper_tests.cu" "${CMAKE_BINARY_DIR}")

ConfigureExecutable(sawtooth_test "${CMAKE_CURRENT_SOURCE_DIR}/src/sawtooth_test.cu" "${CMAKE_BINARY_DIR}")

ConfigureExecutable(tcf_key_val_tests "${CMAKE_CURRENT_SOURCE_DIR}/src/tcf_key_val_tests.cu" "${CMAKE_BINARY_DIR}")