cmake_minimum_required(VERSION 3.2)
project(ddrnet)

message("=========", ${CMAKE_SOURCE_DIR})
message("=========", ${PROJECT_SOURCE_DIR})


file(GLOB TRT_FILES 
    "${CMAKE_SOURCE_DIR}/src/*.h" 
    "${CMAKE_SOURCE_DIR}/src/*.cpp" 
    "${CMAKE_SOURCE_DIR}/src/plugin/*.cu" 
    "${CMAKE_SOURCE_DIR}/src/plugin/*.h"
    "${CMAKE_SOURCE_DIR}/src/plugin/*.cpp")

include_directories(${CMAKE_SOURCE_DIR}/src)

set(SOURCE_FILES main.cpp)
add_executable(ddrnet ${SOURCE_FILES} ${TRT_FILES})


target_link_libraries(ddrnet nvonnxparser)
target_link_libraries(ddrnet nvinfer)
target_link_libraries(ddrnet cudart)
target_link_libraries(ddrnet ${OpenCV_LIBS})