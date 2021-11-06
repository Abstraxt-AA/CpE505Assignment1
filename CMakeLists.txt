cmake_minimum_required(VERSION 3.20)
project(Assignment1)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -march=native")

add_executable(Assignment1 main.cpp)
