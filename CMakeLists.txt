cmake_minimum_required(VERSION 3.17)
project(N_bodies)

set(CMAKE_CXX_STANDARD 14)

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

# added -fopenmp
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fopenmp")

add_executable(N_bodies main.cpp scatter.c openmpi_demo.c main_test.cpp)
target_link_libraries(N_bodies ${MPI_C_LIBRARIES})

set(SOURCE_FILES main.cpp)
