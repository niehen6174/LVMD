cmake_minimum_required(VERSION 2.6)

project(flowWarp)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")


cuda_add_library(gridSamplerPlugin SHARED ${PROJECT_SOURCE_DIR}/gridSamplerPlugin.cpp ${PROJECT_SOURCE_DIR}/gridSampler.cu)
target_link_libraries(gridSamplerPlugin nvinfer cudart)

add_executable(flowWarp  ${PROJECT_SOURCE_DIR}/flow_warp.cpp)
target_link_libraries(flowWarp nvinfer)
target_link_libraries(flowWarp cudart)
target_link_libraries(flowWarp gridSamplerPlugin)


add_definitions(-O2 -pthread)


