cmake_minimum_required(VERSION 3.31.6)
set(CMAKE_MODULE_PATH "/usr/local/cuda-12.6/lib/cmake/CUDA")
project(CheckCUDA CUDA)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/result.cmake"
  "set(CMAKE_CUDA_COMPILER \"${CMAKE_CUDA_COMPILER}\")\n"
  "set(CMAKE_CUDA_HOST_COMPILER \"${CMAKE_CUDA_HOST_COMPILER}\")\n"
  )