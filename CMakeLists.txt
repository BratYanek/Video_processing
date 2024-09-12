project(OpenCVExample)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)

add_executable(camera camera.cpp)
add_executable(training training.cpp)
add_executable(recognition recognition.cpp)

target_link_libraries(camera ${OpenCV_LIBS})
target_link_libraries(training ${OpenCV_LIBS})
target_link_libraries(recognition ${OpenCV_LIBS})

include_directories(${OpenCV_INCLUDE_DIRS})
