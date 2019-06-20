
CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

PROJECT(CUT_PURSUIT_SEG)

set (CMAKE_CXX_STANDARD 11)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules ${CMAKE_MODULE_PATH})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -fopenmp -O3")

##############################
### Find required packages ###
##############################

include_directories(../include)

find_package(PythonLibs)
find_package(PythonInterp)
find_package(NumPy 1.5 REQUIRED)

find_package(Boost 1.65.0 COMPONENTS graph REQUIRED) #system filesystem thread serialization
if (${Boost_MINOR_VERSION} LESS 67 )
    find_package(Boost 1.65.0 COMPONENTS numpy${PYTHON_VERSION_MAJOR} REQUIRED) #system filesystem thread serialization
else()
    set(PYTHONVERSION ${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR})
    find_package(Boost 1.67.0 COMPONENTS numpy${PYTHONVERSION} REQUIRED)
endif()
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})



message("PYTHON LIBRARIES ARE " ${PYTHON_LIBRARIES})
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS} ${PYTHON_NUMPY_INCLUDE_DIR})
LINK_DIRECTORIES(${PYTHON_LIBRARY_DIRS})

file(GLOB CP_HEADERS include/*.h)

set(CMAKE_LD_FLAG "${CMAKE_LD_FLAGS} -shared -Wl -fPIC --export-dynamic -o -O3 -Wall -std=c++11")
message(${Boost_LIBRARIES})
add_library(cp SHARED cutpursuit.cpp ${CP_HEADERS})
target_link_libraries(cp
    ${Boost_LIBRARIES}
    ${PYTHON_LIBRARIES}
    boost_system
    boost_python37
    )