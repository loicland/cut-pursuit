
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(CUT_PURSUIT_SEG)

set (CMAKE_CXX_STANDARD 11)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules ${CMAKE_MODULE_PATH})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -fopenmp -O3")
##############################
### Find required packages ###
##############################

include_directories(include)

find_package(Boost 1.58.0 COMPONENTS graph numpy REQUIRED) #system filesystem thread serialization
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

find_package(PythonLibs)
message("PYTHON LIBRARIES ARE " ${PYTHON_LIBRARIES})
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS})
LINK_DIRECTORIES(${PYTHON_LIBRARY_DIRS})

file(GLOB CP_HEADERS include/*.h)

set(CMAKE_LD_FLAG "${CMAKE_LD_FLAGS} -shared -Wl -fPIC --export-dynamic -O3 -Wall -std=c++11")

add_library(cp SHARED cutpursuit.cpp ${CP_HEADERS})
target_link_libraries(cp
    ${Boost_LIBRARIES}
    ${PYTHON_LIBRARIES}
    )
