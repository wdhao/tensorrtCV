#指定CMake的最小版本要求
cmake_minimum_required(VERSION 3.2)
#定义工程名称
project(ddrnet)

#显示消息
message("=========", ${CMAKE_SOURCE_DIR})#CMAKE_SOURCE_DIR 工程顶层目录，绝对路径 "/home/zxl/mywork/tensorrtCV"
message("=========", ${PROJECT_SOURCE_DIR})

#将文件组成的列表，保存到变量TRT_FILES中
file(GLOB TRT_FILES
    "${CMAKE_SOURCE_DIR}/src/*.h"
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/plugin/*.cu"
    "${CMAKE_SOURCE_DIR}/src/plugin/*.h"
    "${CMAKE_SOURCE_DIR}/src/plugin/*.cpp")

#向工程添加头文件搜索路径
include_directories(${CMAKE_SOURCE_DIR}/src)#"/home/zxl/mywork/tensorrtCV/src"

#显式定义SOURCE_FILES ，其值为main.cpp
set(SOURCE_FILES main.cpp)

#编译生成可执行文件ddrnet
add_executable(ddrnet ${SOURCE_FILES} ${TRT_FILES})

#将目标文件与库文件进行链接
target_link_libraries(ddrnet nvonnxparser)
target_link_libraries(ddrnet nvinfer)
target_link_libraries(ddrnet cudart)
target_link_libraries(ddrnet ${OpenCV_LIBS})
