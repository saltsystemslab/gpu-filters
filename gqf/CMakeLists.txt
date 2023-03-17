cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(gqf VERSION 1.0.0 LANGUAGES CXX CUDA)



###################################################################################################
# - build type ------------------------------------------------------------------------------------

# Set a default build type if none was specified
set(DEFAULT_BUILD_TYPE "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' since none specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()


set(GPU_ARCHS "")
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/EvalGpuArchs.cmake)
  evaluate_gpu_archs(GPU_ARCHS)

find_package(CUDAToolkit 11.0 REQUIRED)
set(CMAKE_CUDA_RUNTIME_LIBRARY SHARED)


## Package Target


include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)

file(GLOB all_SRCS
  "${PROJECT_SOURCE_DIR}/include/*.cuh"
  "${PROJECT_SOURCE_DIR}/include/*.hpp"
  "${PROJECT_SOURCE_DIR}/src/gqf.cu"
  "${PROJECT_SOURCE_DIR}/src/sqf.cu"
  "${PROJECT_SOURCE_DIR}/src/RSQF.cu"
  "${PROJECT_SOURCE_DIR}/src/hashutil.cu"
  "${PROJECT_SOURCE_DIR}/src/partitioned_counter.cu"

)

add_library(gqf STATIC ${all_SRCS})

target_include_directories(gqf PUBLIC ${PROJECT_SOURCE_DIR}/include)

set_target_properties(gqf PROPERTIES CUDA_ARCHITECTURES ${GPU_ARCHS})

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")


#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_EXTENSIONS OFF)
#set(CMAKE_C_COMPILER $ENV{CC})
#set(CMAKE_CXX_COMPILER $ENV{CXX})
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}") 


set_target_properties(gqf PROPERTIES CMAKE_CUDA_RUNTIME_LIBRARY Static)
set_target_properties(gqf PROPERTIES CUDA_SEPARABLE_COMPILATION ON )
