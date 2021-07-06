cmake_minimum_required(VERSION 3.2)
project(yolov5)


file(GLOB TRT_FILES 
    "${CMAKE_SOURCE_DIR}/src/*.h" 
    "${CMAKE_SOURCE_DIR}/src/*.cpp" 
    "${CMAKE_SOURCE_DIR}/src/plugin/*.cu" 
    "${CMAKE_SOURCE_DIR}/src/plugin/*.h"
    "${CMAKE_SOURCE_DIR}/src/plugin/*.cpp")

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/src/plugin)

set(SOURCE_FILES main.cpp)
add_executable(yolov5 ${SOURCE_FILES} ${TRT_FILES})


target_link_libraries(yolov5 nvonnxparser)
target_link_libraries(yolov5 nvinfer)
target_link_libraries(yolov5 cudart)
target_link_libraries(yolov5 ${OpenCV_LIBS})