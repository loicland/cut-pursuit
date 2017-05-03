# Graph for Cut Pursuit
# author: Loic Landrieu
# date: 2017

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(CUT_PURSUIT)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules ${CMAKE_MODULE_PATH})

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -pthread  -Wall -std=c++11")

##############################
### Find required packages ###
##############################

include_directories(include )

find_package(Boost 1.35.0 COMPONENTS graph graph_parallel REQUIRED) #system filesystem thread serialization
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

file(GLOB CP_HEADERS include/*.h)

add_subdirectory(src)

