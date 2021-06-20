cmake_minimum_required(VERSION 3.5.1)
project(libtorch-yolov5)

set(CMAKE_CXX_STANDARD 14)
# It prevents the decay to C++98 when the compiler does not support C++14
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# It disables the use of compiler-specific extensions
# e.g. -std=c++14 rather than -std=gnu++14
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# Try to find OpenCV
#. set(OpenCV_DIR /Users/cxu/opencv-3.4.4/build_xcode)
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message(STATUS "OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}" \n)
else ()
    message(FATAL_ERROR "Could not locate OpenCV" \n)
endif()

set(Torch_DIR /Users/cxu/anaconda3/lib/python3.7/site-packages/torch/share/cmake/Torch)
# /home/cx/anaconda3/envs/torch17/lib/python3.7/site-packages/torch/share/cmake/Torch
find_package(Torch PATHS ${Torch_DIR} NO_DEFAULT REQUIRED)
if (Torch_FOUND)
    message(STATUS "Torch library found!")
    message(STATUS "    include path: ${TORCH_INCLUDE_DIRS}" \n)
else ()
    message(FATAL_ERROR "Could not locate Torch" \n)
endif()

include_directories( ${PROJECT_SOURCE_DIR}/models ${PROJECT_SOURCE_DIR}/utils ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/dependency/yaml-cpp)
link_directories()
link_libraries( ${CMAKE_HOME_DIRECTORY}/dependency/yaml-cpp/libyaml-cpp.a )

file(GLOB SOURCE_FILE models/*.cpp utils/*.cpp)

add_library(common ${SOURCE_FILE})
add_executable(train train.cpp )
add_executable(test test.cpp )
add_executable(detect detect.cpp )

target_link_libraries (
    common
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)

target_link_libraries (train
        ${OpenCV_LIBS}
        ${TORCH_LIBRARIES}
)
target_link_libraries (test
        ${OpenCV_LIBS}
        ${TORCH_LIBRARIES}
)
target_link_libraries (detect
    common
)
