cmake_minimum_required(VERSION 3.2)

project(tensorrtF)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

if(WIN32)
add_definitions(-std=c++11)
set(OpenCV_DIR "D:\\opencv\\opencv346\\build")
endif(WIN32)

if(UNIX)
#find_package(OpenCV)
find_package(OpenCV 3.4.11 REQUIRED)
endif(UNIX)

set(TENSORRT_DIR "/mnt/f/WSL/TensorRT-7.2.3.4/targets/x86_64-linux-gnu")

find_package(CUDA REQUIRED)
enable_language(CUDA)
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")

include_directories(${CUDA_INCLUDE_DIRS})

set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11; -g; -G;-gencode; arch=compute_75;code=sm_75)

enable_language(CUDA)  
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -D_MWAITXINTRIN_H_INCLUDED")

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
include_directories(${TENSORRT_DIR}\\include)

link_directories(${TENSORRT_DIR}\\lib)

find_package(OpenCV QUIET
    NO_MODULE
    NO_DEFAULT_PATH
    NO_CMAKE_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    NO_CMAKE_PACKAGE_REGISTRY
    NO_CMAKE_BUILDS_PATH
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_SYSTEM_PACKAGE_REGISTRY
)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(example)