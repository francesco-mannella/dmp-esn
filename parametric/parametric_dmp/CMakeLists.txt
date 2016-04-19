# States that CMake required version must be greater than 2.6
cmake_minimum_required(VERSION 2.6)
# Project name is not mandatory, but you should use it
project(PARAMETRIC_DMP)
###############################################################################
# General settings



# Never build inside the source tree
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)



# Some flags for CXX
set(CMAKE_CXX_FLAGS "-W -Wall -Werror -std=c++0x -fPIC")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Define macros to check for real time requirements
# To activate the EIGEN_MALLOC asserts use: cmake .. -DREALTIME_CHECKS=1 -DCMAKE_BUILD_TYPE=Debug
if(REALTIME_CHECKS)
   add_definitions(-DREALTIME_CHECKS)
endif()

include_directories(${CMAKE_SOURCE_DIR}/src)
link_directories(${CMAKE_SOURCE_DIR}/lib)


###############################################################################
# Find boost packages
find_package( Boost COMPONENTS filesystem system serialization regex REQUIRED)


###########################################################################à###
# define dmp includes/libs
set(dmp_INCLUDES 
    "/usr/local/include/dmp" 
    "/usr/local/include/dynamicalsystems" 
    "/usr/local/include/functionapproximators" 
    )

set(dmp_LIBS 
    "-l dmp" 
    "-l dynamicalsystems" 
    "-l functionapproximators" 
    "-l lwpr"
    )



###########################################################################
# altogether



set( INCLUDES 
    ${Boost_INCLUDES}
    ${dmp_INCLUDES}
    )

set( LIBS 
    ${Boost_LIBRARIES} 
    ${dmp_LIBS}
    )


###############################################################################
# Things to compile are in the src/ directory
add_subdirectory(src)


