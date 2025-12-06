# swatbotics/apriltags-cpp/CMakeLists.txt

cmake_minimum_required(VERSION 2.6)

project(APRILTAGS)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

if(APPLE)
  include_directories(/opt/local/include) # MacPorts
  link_directories(/opt/local/lib)
  find_library(OPENGL_LIBRARY OpenGL)
else()
  find_library(OPENGL_LIBRARY GL)
  find_library(GLU_LIBRARY GLU)
  set(OPENGL_LIBRARY ${OPENGL_LIBRARY} ${GLU_LIBRARY})
endif()

find_library(GLUT_LIBRARY glut)

include(FindPkgConfig)

# DEPENDENCIES #################################################################
if(WIN32)
  # On Windows, we rely on env variables to find install paths
  if(NOT DEFINED ENV{OpenCV_DIR})
    message(FATAL_ERROR "OpenCV_DIR env var not set, please set it to the OpenCV install directory")
  endif()
  set(OpenCV_DIR "$ENV{OpenCV_DIR}\\build" CACHE PATH "OpenCV build directory" FORCE)
endif()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_definitions(-DNOMINMAX -DHAVE_OPENCV)
#pkg_search_module(OPENCV REQUIRED opencv>=2.3 opencv-2.3.1)
#include_directories(${OPENCV_INCLUDE_DIRS})

pkg_check_modules(CAIRO cairo)

if (${CAIRO_FOUND})
add_definitions(-DMZ_HAVE_CAIRO)
endif ()

find_package( CGAL QUIET COMPONENTS  )

if (CGAL_FOUND)
  include( ${CGAL_USE_FILE} )
  add_definitions(-DHAVE_CGAL)
  find_package( Boost REQUIRED )
else()
  message("CGAL not found, can not use new quad detector")
endif()

if (APPLE OR UNIX)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wsign-compare -frounding-math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wsign-compare -frounding-math")
endif()

if(WIN32)
  set( CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}" ${CMAKE_MODULE_PATH} )
endif()

if(WIN32)
  find_package(k4a)
endif()


add_subdirectory(src)

