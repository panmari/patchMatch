cmake_minimum_required(VERSION 3.3)
project(patchMatch)

set(OpenCV_DIR /home/moser/opencv)

find_package( OpenCV REQUIRED )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp)
add_executable(patchMatch main.cpp ExhaustivePatchMatch.cpp ExhaustivePatchMatch.h)
target_link_libraries( patchMatch ${OpenCV_LIBS} )

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")