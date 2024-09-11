project(OpenCVExample)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
add_executable(camera camera.cpp)
target_link_libraries(camera ${OpenCV_LIBS})
include_directories(${OpenCV_INCLUDE_DIRS})
