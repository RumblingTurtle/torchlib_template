cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(torchlib_template)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "-g -pthread") 
set(CMAKE_PREFIX_PATH CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/libtorch)

find_package(Torch REQUIRED)

set(SOURCES src/main.cpp)

add_executable(torchlib_template ${SOURCES})
target_include_directories(torchlib_template PUBLIC include)
target_link_libraries(torchlib_template ${TORCH_LIBRARIES})
