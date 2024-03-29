cmake_minimum_required(VERSION 2.6)

project(real-esrgan)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

if(WIN32)
enable_language(CUDA)
endif(WIN32)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -g -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
cuda_add_library(myplugins SHARED preprocess.cu postprocess.cu)
target_link_libraries(myplugins nvinfer cudart)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

cuda_add_executable(real-esrgan real-esrgan.cpp)

target_link_libraries(real-esrgan nvinfer)
target_link_libraries(real-esrgan cudart)
target_link_libraries(real-esrgan myplugins)
target_link_libraries(real-esrgan ${OpenCV_LIBS})

if(UNIX)
add_definitions(-O2 -pthread)
endif(UNIX)


