cmake_minimum_required(VERSION 3.10)
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
target_link_libraries(patch_match_lib ${OpenCV_LIBS} ${Boost_LIBRARIES})

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
find_package(GTest REQUIRED)

macro(package_add_test TESTNAME)
    # create an exectuable in which the tests will be stored
    add_executable(${TESTNAME} ${ARGN})
    # link the Google test infrastructure, mocking library, and a default main fuction to
    # the test executable.  Remove g_test_main if writing your own main function.
    target_link_libraries(${TESTNAME} gtest gtest_main patch_match_lib)
    # gtest_discover_tests replaces gtest_add_tests,
    # see https://cmake.org/cmake/help/v3.10/module/GoogleTest.html for more options to pass to it
    gtest_discover_tests(${TESTNAME}
            # set a working directory so your project root so that you can find test data via paths relative to the project root
            WORKING_DIRECTORY ${PROJECT_DIR}
            PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_DIR}"
            )
    set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
endmacro()

file(GLOB test_source_files tests/*.cpp)
package_add_test(runUnitTests ${test_source_files})
