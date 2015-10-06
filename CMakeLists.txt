cmake_minimum_required(VERSION 3.3)
project(patchMatch)

set(OpenCV_DIR /home/moser/opencv)

find_package( OpenCV REQUIRED )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES src/main.cpp)
add_executable(patchMatch src/main.cpp src/ExhaustivePatchMatch.cpp src/ExhaustivePatchMatch.h src/RandomizedPatchMatch.cpp src/RandomizedPatchMatch.h src/TrivialReconstruction.cpp src/TrivialReconstruction.h)
target_link_libraries( patchMatch ${OpenCV_LIBS} )

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")