
#file(GLOB SRC_LIST "*.cpp")
file(GLOB INC_LIST "*.h")
include_directories(.)
add_library(apriltags ${INC_LIST} CameraUtil.cpp
  DebugImage.cpp
  Geometry.cpp 
  GrayModel.cpp
  MathUtil.cpp 
  Refine.cpp
  TagDetector.cpp 
  TagFamily.cpp
  TagFamilies.cpp
  UnionFindSimple.cpp
  getopt.cpp)

set(AT_LIBS apriltags ${OpenCV_LIBS})

target_link_libraries(apriltags ${AT_LIBS})

add_executable(azure_apriltags azure_apriltags.cpp transformation_helpers.cpp)
target_link_libraries(azure_apriltags ${AT_LIBS} k4a::k4a k4a::k4arecord)