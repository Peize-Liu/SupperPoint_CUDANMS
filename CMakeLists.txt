cmake_minimum_required(VERSION 3.0.0)
project(supper_point VERSION 0.1.0 LANGUAGES C CXX)


find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
  /usr/local/include/
  ${CUDA_INCLUDE_DIRS}
)

add_executable(supper_point ./main.cpp)


target_link_libraries(supper_point
  nvinfer
  nvinfer_plugin
  ${OpenCV_LIBRARIES}
  ${CUDA_LIBRARIES}
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
