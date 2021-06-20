cmake_minimum_required(VERSION 3.5.1)
project(yaml)

set(CMAKE_CXX_STANDARD 14)
# It prevents the decay to C++98 when the compiler does not support C++14
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# It disables the use of compiler-specific extensions
# e.g. -std=c++14 rather than -std=gnu++14
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")


set(Torch_DIR /Users/cxu/anaconda3/lib/python3.7/site-packages/torch/share/cmake/Torch)
# /home/cx/anaconda3/envs/torch17/lib/python3.7/site-packages/torch/share/cmake/Torch
find_package(Torch PATHS ${Torch_DIR} NO_DEFAULT REQUIRED)
if (Torch_FOUND)
    message(STATUS "Torch library found!")
    message(STATUS "    include path: ${TORCH_INCLUDE_DIRS}" \n)
else ()
    message(FATAL_ERROR "Could not locate Torch" \n)
endif()

include_directories( ${PROJECT_SOURCE_DIR}/ ${PROJECT_SOURCE_DIR}/ ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/yaml-cpp "/Users/cxu/Desktop/libyolov5/utils")
link_directories()
link_libraries( ${CMAKE_HOME_DIRECTORY}/yaml-cpp/libyaml-cpp.a )

file(GLOB SOURCE_FILE *.cpp)

add_executable(dp dp.cpp ${SOURCE_FILE})

target_link_libraries (
    dp
    ${TORCH_LIBRARIES}
)