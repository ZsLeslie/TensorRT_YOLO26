cmake_minimum_required(VERSION 3.10)

project(yolo26_seg)

# Keep FindCUDA available with CMake 3.27 and newer.
if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE release)

# ============= cuda ============
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# ============= tensorrt ============
set(TENSORRT_ROOT "" CACHE PATH "TensorRT 8.x installation directory")
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} $ENV{TENSORRT_ROOT}
  PATH_SUFFIXES include include/x86_64-linux-gnu include/aarch64-linux-gnu)
find_library(TENSORRT_LIBRARY nvinfer
  HINTS ${TENSORRT_ROOT} $ENV{TENSORRT_ROOT}
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu)
find_library(TENSORRT_ONNX_LIBRARY nvonnxparser
  HINTS ${TENSORRT_ROOT} $ENV{TENSORRT_ROOT}
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu)
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY OR NOT TENSORRT_ONNX_LIBRARY)
  message(FATAL_ERROR "TensorRT not found. Set -DTENSORRT_ROOT=/path/to/TensorRT-8.x")
endif()
include_directories(${TENSORRT_INCLUDE_DIR})

# ============ opencv ============
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ============ yolo infer shared lib ============
include_directories(${PROJECT_SOURCE_DIR}/include)
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)
cuda_add_library(yolo_infer SHARED ${SRCS})
target_link_libraries(yolo_infer ${CUDA_CUDART_LIBRARY} ${TENSORRT_LIBRARY} ${TENSORRT_ONNX_LIBRARY} ${OpenCV_LIBS})

# ======== main execute file ========
add_executable(main ${PROJECT_SOURCE_DIR}/main.cpp)
target_link_libraries(main yolo_infer)
