
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

if(k4a_FOUND AND k4arecord_FOUND AND NOT SKIP_KINECT_AZURE)
  add_executable(camera_apriltags camera_apriltags.cpp azure_kinect_helpers.cpp)
  target_link_libraries(camera_apriltags ${AT_LIBS} k4a::k4a k4a::k4arecord nlohmann_json::nlohmann_json)
  target_compile_definitions(camera_apriltags PRIVATE KINECT_AZURE_LIBS)
  target_compile_features(camera_apriltags PRIVATE cxx_std_17)
else()
  add_executable(camera_apriltags camera_apriltags.cpp)
  target_link_libraries(camera_apriltags ${AT_LIBS} nlohmann_json::nlohmann_json)
  target_compile_features(camera_apriltags PRIVATE cxx_std_17)
endif()