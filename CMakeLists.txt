cmake_minimum_required(VERSION 2.8)
project(patchMatch)

set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost REQUIRED)
include_directories( ${Boost_INCLUDE_DIRS} )

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


file(GLOB patch_match_source src/*(.cpp|.h) src/patch_match_provider/*)
if(OpenCV_CUDA_VERSION)
    add_definitions(-DOpenCV_CUDA_VERSION=${OpenCV_CUDA_VERSION})
    file(GLOB additional_patch_match_source src/patch_match_provider/gpu/*)
else()
    file(GLOB additional_patch_match_source src/patch_match_provider/cpu/*)
endif()
LIST(APPEND patch_match_source ${additional_patch_match_source})


add_library(patch_match_lib ${patch_match_source})
target_link_libraries(patch_match_lib ${OpenCV_LIBS} ${Boost_LIBRARIES} profiler)

add_executable(reconstruction src/main_reconstruction.cxx)
target_link_libraries(reconstruction patch_match_lib)

add_executable(hole_filling src/main_hole_filling.cxx)
target_link_libraries(hole_filling patch_match_lib)

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
