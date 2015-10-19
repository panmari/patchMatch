cmake_minimum_required(VERSION 2.8)
project(patchMatch)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost REQUIRED)
include_directories( ${Boost_INCLUDE_DIRS} )

find_package(CUDA)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --gpu-architecture sm_30)
file(GLOB_RECURSE poisson_solver_src src/poisson_solver/*)
CUDA_ADD_LIBRARY(poisson_solver_lib ${poisson_solver_src})

find_package(OpenCV REQUIRED)
file(GLOB_RECURSE patch_match_source src/*(.cpp|.h))
add_library(patch_match_lib ${patch_match_source})
target_link_libraries(patch_match_lib poisson_solver_lib ${OpenCV_LIBS} ${Boost_LIBRARIES})

add_executable(reconstruction src/main_reconstruction.cxx)
target_link_libraries(reconstruction patch_match_lib)

add_executable(hole_filling src/main_hole_filling.cxx)
target_link_libraries(hole_filling patch_match_lib)

add_executable(seamless_clone src/main_seamless_clone.cxx)
target_link_libraries(seamless_clone patch_match_lib)

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
target_link_libraries(runUnitTests patch_match_lib opencv_ts)
add_test( runUnitTests runUnitTests )
