cmake_minimum_required(VERSION 3.10)
project(VideoProcessor VERSION 1.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set OpenCV_CUDA_VERSION to avoid the error
set(OpenCV_STATIC OFF)
set(OpenCV_CUDA_VERSION "")

# Find OpenCV package
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Optional CUDA support - using modern approach
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_definitions(-DWITH_CUDA)
    message(STATUS "CUDA found, enabling GPU acceleration")
else()
    message(STATUS "CUDA not found, falling back to CPU implementation")
endif()

# Add executable
add_executable(video_processor 
    src/main.cpp
    src/camera.cpp
    src/timer.cpp
)

# Link libraries
target_link_libraries(video_processor ${OpenCV_LIBS})

# Link CUDA if available
if(CMAKE_CUDA_COMPILER)
    # For CUDA-specific libraries if needed
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        target_link_libraries(video_processor CUDA::cudart)
    endif()
endif()

# Output binary to bin directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)