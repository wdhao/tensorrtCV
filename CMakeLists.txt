#指定CMake的最小版本要求
cmake_minimum_required(VERSION 3.2)

#定义工程名称
project(tensorrtF)

#指定C++版本11
set(CMAKE_CXX_STANDARD 11)

#设置g++编译选项 使用c++11 -O3优化等级
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

#CUDA设置
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

#win32环境设置opencv依赖
if(WIN32)
add_definitions(-std=c++11)
set(OpenCV_DIR "D:\\opencv\\opencv346\\build")
endif(WIN32)

#UNIX环境设置opencv依赖版本
if(UNIX)
find_package(OpenCV 3.4.3 REQUIRED)#查找opencv依赖，REQUIRED可选字段。表示一定要找到包，找不到的话就立即停掉整个cmake。而如果不指定REQUIRED则cmake会继续执行。
endif(UNIX)

#显式定义tensorrt目录
set(TENSORRT_DIR "/home/zxl/TensorRT-7.2.3.4/targets/x86_64-linux-gnu")

#寻找CUDA依赖
find_package(CUDA REQUIRED)

#支持CUDA语言
enable_language(CUDA)
#输出正常 STATUS
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")

#向工程添加CUDA头文件搜索路径
include_directories(${CUDA_INCLUDE_DIRS})

#设置nvcc编译选项
set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11; -g; -G;-gencode; arch=compute_86;code=sm_86)#原来是75

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -D_MWAITXINTRIN_H_INCLUDED")
#-Wall选项意思是编译后显示所有警告
#-Ofast将不会严格遵循语言标准，除了启用所有的-O3优化选项之外，也会针对某些语言启用部分优化

#向工程添加tensorrt的头文件搜索路径
include_directories(${TENSORRT_DIR}/include)#原来是\\
#向工程添加tensorrt的库文件搜索路径
link_directories(${TENSORRT_DIR}/lib)#原来是\\

#查找opencv依赖
find_package(OpenCV QUIET  #QUIET表示disable信息输出，即命令运行时不输出相关message
    NO_MODULE #cmake不在MODULE模式的路径下查找包，直接进入CONFIG模式
    NO_DEFAULT_PATH
    NO_CMAKE_PATH#关闭cmake特定的缓存变量这一查找顺序
    NO_CMAKE_ENVIRONMENT_PATH#跳过搜索cmake特定的环境变量
    NO_SYSTEM_ENVIRONMENT_PATH#跳过搜索标准的系统环境变量PATH
    NO_CMAKE_PACKAGE_REGISTRY#跳过搜索存储在cmake的"User Package Registry"(用户包注册表)中的路径
    NO_CMAKE_BUILDS_PATH
    NO_CMAKE_SYSTEM_PATH#通过设定当前系统定义的cmake变量
    NO_CMAKE_SYSTEM_PACKAGE_REGISTRY#跳过在cmake的"System Package Registry"(系统包注册表）中查找
)

#输出正常 STATUS
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

#向工程添加opencv的头文件搜索路径
include_directories(${OpenCV_INCLUDE_DIRS})

#对子文件夹项目进行cmake编译
add_subdirectory(example)
