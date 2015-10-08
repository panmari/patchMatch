cmake_minimum_required(VERSION 2.8)
project(patchMatch)

find_package( OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

file(GLOB_RECURSE patch_match_source src/*)
add_library(patch_match_lib SHARED ${patch_match_source})
target_link_libraries(patch_match_lib ${OpenCV_LIBS} )

set(SOURCE_FILES src/main.cpp)
add_executable(pach_match src/main.cpp)
target_link_libraries(pach_match patch_match_lib )

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")


################################
# Unit Tests
################################
# OpenCV also includes gtest, we don't need to add another dependency.
enable_testing()
# Add test cpp files
file(GLOB test_source_files tests/*.cpp)
add_executable(runUnitTests ${test_source_files})
# Link test executable against gtest, gtest_main and other used libraries.
target_link_libraries(runUnitTests patch_match_lib ${OpenCV_LIBS} opencv_ts)
add_test( runUnitTests runUnitTests )
