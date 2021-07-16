#指定CMake的最小版本要求
cmake_minimum_required(VERSION 3.2)
#对子文件夹项目进行cmake编译
add_subdirectory(ddrnet)
#对子文件夹项目进行cmake编译
add_subdirectory(yolov5)
