cmake_minimum_required(VERSION 3.15)
project(PSO)

set(CMAKE_CXX_STANDARD 14)

find_package (Eigen3  REQUIRED)
find_package(Python2 COMPONENTS Development NumPy)

add_executable(PSO main.cpp)
include_directories( "/usr/include/eigen3" )

target_include_directories(PSO PRIVATE ${Python2_INCLUDE_DIRS} ${Python2_NumPy_INCLUDE_DIRS})
target_link_libraries(PSO Python2::Python Python2::NumPy)