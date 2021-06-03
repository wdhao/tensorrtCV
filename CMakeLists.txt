cmake_minimum_required(VERSION 3.4)

project(tensorrtF)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

if(WIN32)
add_definitions(-std=c++11)
set(OpenCV_DIR "D:\\opencv\\opencv346\\build")
endif(WIN32)

if(UNIX)
find_package(OpenCV)
endif(UNIX)

set(TENSORRT_DIR "D:\\TensorRT-7.0.0.11.Windows10.x86_64.cuda-10.2.cudnn7.6\\TensorRT-7.0.0.11")

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

file(GLOB SOURCE_FILES "*.h" "*.cpp" "*.cu")
add_executable(tensorrtF ${SOURCE_FILES})

target_link_libraries(tensorrtF nvonnxparser)
target_link_libraries(tensorrtF nvinfer)
target_link_libraries(tensorrtF cudart)
target_link_libraries(tensorrtF ${OpenCV_LIBS})

if(UNIX)
add_definitions(-O2)
endif(UNIX)