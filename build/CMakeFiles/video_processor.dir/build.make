# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.31

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/skand/miniconda3/lib/python3.12/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/skand/miniconda3/lib/python3.12/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/skand/video-processor

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/skand/video-processor/build

# Include any dependencies generated for this target.
include CMakeFiles/video_processor.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/video_processor.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/video_processor.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/video_processor.dir/flags.make

CMakeFiles/video_processor.dir/codegen:
.PHONY : CMakeFiles/video_processor.dir/codegen

CMakeFiles/video_processor.dir/src/main.cpp.o: CMakeFiles/video_processor.dir/flags.make
CMakeFiles/video_processor.dir/src/main.cpp.o: /home/skand/video-processor/src/main.cpp
CMakeFiles/video_processor.dir/src/main.cpp.o: CMakeFiles/video_processor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/skand/video-processor/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/video_processor.dir/src/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/video_processor.dir/src/main.cpp.o -MF CMakeFiles/video_processor.dir/src/main.cpp.o.d -o CMakeFiles/video_processor.dir/src/main.cpp.o -c /home/skand/video-processor/src/main.cpp

CMakeFiles/video_processor.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/video_processor.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/skand/video-processor/src/main.cpp > CMakeFiles/video_processor.dir/src/main.cpp.i

CMakeFiles/video_processor.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/video_processor.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/skand/video-processor/src/main.cpp -o CMakeFiles/video_processor.dir/src/main.cpp.s

CMakeFiles/video_processor.dir/src/camera.cpp.o: CMakeFiles/video_processor.dir/flags.make
CMakeFiles/video_processor.dir/src/camera.cpp.o: /home/skand/video-processor/src/camera.cpp
CMakeFiles/video_processor.dir/src/camera.cpp.o: CMakeFiles/video_processor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/skand/video-processor/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/video_processor.dir/src/camera.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/video_processor.dir/src/camera.cpp.o -MF CMakeFiles/video_processor.dir/src/camera.cpp.o.d -o CMakeFiles/video_processor.dir/src/camera.cpp.o -c /home/skand/video-processor/src/camera.cpp

CMakeFiles/video_processor.dir/src/camera.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/video_processor.dir/src/camera.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/skand/video-processor/src/camera.cpp > CMakeFiles/video_processor.dir/src/camera.cpp.i

CMakeFiles/video_processor.dir/src/camera.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/video_processor.dir/src/camera.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/skand/video-processor/src/camera.cpp -o CMakeFiles/video_processor.dir/src/camera.cpp.s

CMakeFiles/video_processor.dir/src/timer.cpp.o: CMakeFiles/video_processor.dir/flags.make
CMakeFiles/video_processor.dir/src/timer.cpp.o: /home/skand/video-processor/src/timer.cpp
CMakeFiles/video_processor.dir/src/timer.cpp.o: CMakeFiles/video_processor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/skand/video-processor/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/video_processor.dir/src/timer.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/video_processor.dir/src/timer.cpp.o -MF CMakeFiles/video_processor.dir/src/timer.cpp.o.d -o CMakeFiles/video_processor.dir/src/timer.cpp.o -c /home/skand/video-processor/src/timer.cpp

CMakeFiles/video_processor.dir/src/timer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/video_processor.dir/src/timer.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/skand/video-processor/src/timer.cpp > CMakeFiles/video_processor.dir/src/timer.cpp.i

CMakeFiles/video_processor.dir/src/timer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/video_processor.dir/src/timer.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/skand/video-processor/src/timer.cpp -o CMakeFiles/video_processor.dir/src/timer.cpp.s

CMakeFiles/video_processor.dir/src/frame_buffer.cpp.o: CMakeFiles/video_processor.dir/flags.make
CMakeFiles/video_processor.dir/src/frame_buffer.cpp.o: /home/skand/video-processor/src/frame_buffer.cpp
CMakeFiles/video_processor.dir/src/frame_buffer.cpp.o: CMakeFiles/video_processor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/skand/video-processor/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/video_processor.dir/src/frame_buffer.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/video_processor.dir/src/frame_buffer.cpp.o -MF CMakeFiles/video_processor.dir/src/frame_buffer.cpp.o.d -o CMakeFiles/video_processor.dir/src/frame_buffer.cpp.o -c /home/skand/video-processor/src/frame_buffer.cpp

CMakeFiles/video_processor.dir/src/frame_buffer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/video_processor.dir/src/frame_buffer.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/skand/video-processor/src/frame_buffer.cpp > CMakeFiles/video_processor.dir/src/frame_buffer.cpp.i

CMakeFiles/video_processor.dir/src/frame_buffer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/video_processor.dir/src/frame_buffer.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/skand/video-processor/src/frame_buffer.cpp -o CMakeFiles/video_processor.dir/src/frame_buffer.cpp.s

CMakeFiles/video_processor.dir/src/upscaler.cpp.o: CMakeFiles/video_processor.dir/flags.make
CMakeFiles/video_processor.dir/src/upscaler.cpp.o: /home/skand/video-processor/src/upscaler.cpp
CMakeFiles/video_processor.dir/src/upscaler.cpp.o: CMakeFiles/video_processor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/skand/video-processor/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/video_processor.dir/src/upscaler.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/video_processor.dir/src/upscaler.cpp.o -MF CMakeFiles/video_processor.dir/src/upscaler.cpp.o.d -o CMakeFiles/video_processor.dir/src/upscaler.cpp.o -c /home/skand/video-processor/src/upscaler.cpp

CMakeFiles/video_processor.dir/src/upscaler.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/video_processor.dir/src/upscaler.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/skand/video-processor/src/upscaler.cpp > CMakeFiles/video_processor.dir/src/upscaler.cpp.i

CMakeFiles/video_processor.dir/src/upscaler.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/video_processor.dir/src/upscaler.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/skand/video-processor/src/upscaler.cpp -o CMakeFiles/video_processor.dir/src/upscaler.cpp.s

CMakeFiles/video_processor.dir/src/dnn_super_res.cpp.o: CMakeFiles/video_processor.dir/flags.make
CMakeFiles/video_processor.dir/src/dnn_super_res.cpp.o: /home/skand/video-processor/src/dnn_super_res.cpp
CMakeFiles/video_processor.dir/src/dnn_super_res.cpp.o: CMakeFiles/video_processor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/skand/video-processor/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/video_processor.dir/src/dnn_super_res.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/video_processor.dir/src/dnn_super_res.cpp.o -MF CMakeFiles/video_processor.dir/src/dnn_super_res.cpp.o.d -o CMakeFiles/video_processor.dir/src/dnn_super_res.cpp.o -c /home/skand/video-processor/src/dnn_super_res.cpp

CMakeFiles/video_processor.dir/src/dnn_super_res.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/video_processor.dir/src/dnn_super_res.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/skand/video-processor/src/dnn_super_res.cpp > CMakeFiles/video_processor.dir/src/dnn_super_res.cpp.i

CMakeFiles/video_processor.dir/src/dnn_super_res.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/video_processor.dir/src/dnn_super_res.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/skand/video-processor/src/dnn_super_res.cpp -o CMakeFiles/video_processor.dir/src/dnn_super_res.cpp.s

CMakeFiles/video_processor.dir/src/temporal_consistency.cpp.o: CMakeFiles/video_processor.dir/flags.make
CMakeFiles/video_processor.dir/src/temporal_consistency.cpp.o: /home/skand/video-processor/src/temporal_consistency.cpp
CMakeFiles/video_processor.dir/src/temporal_consistency.cpp.o: CMakeFiles/video_processor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/skand/video-processor/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/video_processor.dir/src/temporal_consistency.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/video_processor.dir/src/temporal_consistency.cpp.o -MF CMakeFiles/video_processor.dir/src/temporal_consistency.cpp.o.d -o CMakeFiles/video_processor.dir/src/temporal_consistency.cpp.o -c /home/skand/video-processor/src/temporal_consistency.cpp

CMakeFiles/video_processor.dir/src/temporal_consistency.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/video_processor.dir/src/temporal_consistency.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/skand/video-processor/src/temporal_consistency.cpp > CMakeFiles/video_processor.dir/src/temporal_consistency.cpp.i

CMakeFiles/video_processor.dir/src/temporal_consistency.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/video_processor.dir/src/temporal_consistency.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/skand/video-processor/src/temporal_consistency.cpp -o CMakeFiles/video_processor.dir/src/temporal_consistency.cpp.s

CMakeFiles/video_processor.dir/src/adaptive_sharpening.cpp.o: CMakeFiles/video_processor.dir/flags.make
CMakeFiles/video_processor.dir/src/adaptive_sharpening.cpp.o: /home/skand/video-processor/src/adaptive_sharpening.cpp
CMakeFiles/video_processor.dir/src/adaptive_sharpening.cpp.o: CMakeFiles/video_processor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/skand/video-processor/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/video_processor.dir/src/adaptive_sharpening.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/video_processor.dir/src/adaptive_sharpening.cpp.o -MF CMakeFiles/video_processor.dir/src/adaptive_sharpening.cpp.o.d -o CMakeFiles/video_processor.dir/src/adaptive_sharpening.cpp.o -c /home/skand/video-processor/src/adaptive_sharpening.cpp

CMakeFiles/video_processor.dir/src/adaptive_sharpening.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/video_processor.dir/src/adaptive_sharpening.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/skand/video-processor/src/adaptive_sharpening.cpp > CMakeFiles/video_processor.dir/src/adaptive_sharpening.cpp.i

CMakeFiles/video_processor.dir/src/adaptive_sharpening.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/video_processor.dir/src/adaptive_sharpening.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/skand/video-processor/src/adaptive_sharpening.cpp -o CMakeFiles/video_processor.dir/src/adaptive_sharpening.cpp.s

CMakeFiles/video_processor.dir/src/selective_bilateral.cpp.o: CMakeFiles/video_processor.dir/flags.make
CMakeFiles/video_processor.dir/src/selective_bilateral.cpp.o: /home/skand/video-processor/src/selective_bilateral.cpp
CMakeFiles/video_processor.dir/src/selective_bilateral.cpp.o: CMakeFiles/video_processor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/skand/video-processor/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/video_processor.dir/src/selective_bilateral.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/video_processor.dir/src/selective_bilateral.cpp.o -MF CMakeFiles/video_processor.dir/src/selective_bilateral.cpp.o.d -o CMakeFiles/video_processor.dir/src/selective_bilateral.cpp.o -c /home/skand/video-processor/src/selective_bilateral.cpp

CMakeFiles/video_processor.dir/src/selective_bilateral.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/video_processor.dir/src/selective_bilateral.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/skand/video-processor/src/selective_bilateral.cpp > CMakeFiles/video_processor.dir/src/selective_bilateral.cpp.i

CMakeFiles/video_processor.dir/src/selective_bilateral.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/video_processor.dir/src/selective_bilateral.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/skand/video-processor/src/selective_bilateral.cpp -o CMakeFiles/video_processor.dir/src/selective_bilateral.cpp.s

CMakeFiles/video_processor.dir/src/display.cpp.o: CMakeFiles/video_processor.dir/flags.make
CMakeFiles/video_processor.dir/src/display.cpp.o: /home/skand/video-processor/src/display.cpp
CMakeFiles/video_processor.dir/src/display.cpp.o: CMakeFiles/video_processor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/skand/video-processor/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object CMakeFiles/video_processor.dir/src/display.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/video_processor.dir/src/display.cpp.o -MF CMakeFiles/video_processor.dir/src/display.cpp.o.d -o CMakeFiles/video_processor.dir/src/display.cpp.o -c /home/skand/video-processor/src/display.cpp

CMakeFiles/video_processor.dir/src/display.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/video_processor.dir/src/display.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/skand/video-processor/src/display.cpp > CMakeFiles/video_processor.dir/src/display.cpp.i

CMakeFiles/video_processor.dir/src/display.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/video_processor.dir/src/display.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/skand/video-processor/src/display.cpp -o CMakeFiles/video_processor.dir/src/display.cpp.s

CMakeFiles/video_processor.dir/src/processor.cpp.o: CMakeFiles/video_processor.dir/flags.make
CMakeFiles/video_processor.dir/src/processor.cpp.o: /home/skand/video-processor/src/processor.cpp
CMakeFiles/video_processor.dir/src/processor.cpp.o: CMakeFiles/video_processor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/skand/video-processor/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object CMakeFiles/video_processor.dir/src/processor.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/video_processor.dir/src/processor.cpp.o -MF CMakeFiles/video_processor.dir/src/processor.cpp.o.d -o CMakeFiles/video_processor.dir/src/processor.cpp.o -c /home/skand/video-processor/src/processor.cpp

CMakeFiles/video_processor.dir/src/processor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/video_processor.dir/src/processor.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/skand/video-processor/src/processor.cpp > CMakeFiles/video_processor.dir/src/processor.cpp.i

CMakeFiles/video_processor.dir/src/processor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/video_processor.dir/src/processor.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/skand/video-processor/src/processor.cpp -o CMakeFiles/video_processor.dir/src/processor.cpp.s

CMakeFiles/video_processor.dir/src/pipeline.cpp.o: CMakeFiles/video_processor.dir/flags.make
CMakeFiles/video_processor.dir/src/pipeline.cpp.o: /home/skand/video-processor/src/pipeline.cpp
CMakeFiles/video_processor.dir/src/pipeline.cpp.o: CMakeFiles/video_processor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/skand/video-processor/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object CMakeFiles/video_processor.dir/src/pipeline.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/video_processor.dir/src/pipeline.cpp.o -MF CMakeFiles/video_processor.dir/src/pipeline.cpp.o.d -o CMakeFiles/video_processor.dir/src/pipeline.cpp.o -c /home/skand/video-processor/src/pipeline.cpp

CMakeFiles/video_processor.dir/src/pipeline.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/video_processor.dir/src/pipeline.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/skand/video-processor/src/pipeline.cpp > CMakeFiles/video_processor.dir/src/pipeline.cpp.i

CMakeFiles/video_processor.dir/src/pipeline.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/video_processor.dir/src/pipeline.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/skand/video-processor/src/pipeline.cpp -o CMakeFiles/video_processor.dir/src/pipeline.cpp.s

# Object files for target video_processor
video_processor_OBJECTS = \
"CMakeFiles/video_processor.dir/src/main.cpp.o" \
"CMakeFiles/video_processor.dir/src/camera.cpp.o" \
"CMakeFiles/video_processor.dir/src/timer.cpp.o" \
"CMakeFiles/video_processor.dir/src/frame_buffer.cpp.o" \
"CMakeFiles/video_processor.dir/src/upscaler.cpp.o" \
"CMakeFiles/video_processor.dir/src/dnn_super_res.cpp.o" \
"CMakeFiles/video_processor.dir/src/temporal_consistency.cpp.o" \
"CMakeFiles/video_processor.dir/src/adaptive_sharpening.cpp.o" \
"CMakeFiles/video_processor.dir/src/selective_bilateral.cpp.o" \
"CMakeFiles/video_processor.dir/src/display.cpp.o" \
"CMakeFiles/video_processor.dir/src/processor.cpp.o" \
"CMakeFiles/video_processor.dir/src/pipeline.cpp.o"

# External object files for target video_processor
video_processor_EXTERNAL_OBJECTS =

/home/skand/video-processor/bin/video_processor: CMakeFiles/video_processor.dir/src/main.cpp.o
/home/skand/video-processor/bin/video_processor: CMakeFiles/video_processor.dir/src/camera.cpp.o
/home/skand/video-processor/bin/video_processor: CMakeFiles/video_processor.dir/src/timer.cpp.o
/home/skand/video-processor/bin/video_processor: CMakeFiles/video_processor.dir/src/frame_buffer.cpp.o
/home/skand/video-processor/bin/video_processor: CMakeFiles/video_processor.dir/src/upscaler.cpp.o
/home/skand/video-processor/bin/video_processor: CMakeFiles/video_processor.dir/src/dnn_super_res.cpp.o
/home/skand/video-processor/bin/video_processor: CMakeFiles/video_processor.dir/src/temporal_consistency.cpp.o
/home/skand/video-processor/bin/video_processor: CMakeFiles/video_processor.dir/src/adaptive_sharpening.cpp.o
/home/skand/video-processor/bin/video_processor: CMakeFiles/video_processor.dir/src/selective_bilateral.cpp.o
/home/skand/video-processor/bin/video_processor: CMakeFiles/video_processor.dir/src/display.cpp.o
/home/skand/video-processor/bin/video_processor: CMakeFiles/video_processor.dir/src/processor.cpp.o
/home/skand/video-processor/bin/video_processor: CMakeFiles/video_processor.dir/src/pipeline.cpp.o
/home/skand/video-processor/bin/video_processor: CMakeFiles/video_processor.dir/build.make
/home/skand/video-processor/bin/video_processor: CMakeFiles/video_processor.dir/compiler_depend.ts
/home/skand/video-processor/bin/video_processor: /usr/local/lib/libopencv_highgui.so.4.12.0
/home/skand/video-processor/bin/video_processor: /usr/local/lib/libopencv_videoio.so.4.12.0
/home/skand/video-processor/bin/video_processor: /usr/local/lib/libopencv_cudaoptflow.so.4.12.0
/home/skand/video-processor/bin/video_processor: /usr/local/lib/libopencv_optflow.so.4.12.0
/home/skand/video-processor/bin/video_processor: /usr/local/lib/libopencv_cudawarping.so.4.12.0
/home/skand/video-processor/bin/video_processor: /usr/local/lib/libopencv_dnn_superres.so.4.12.0
/home/skand/video-processor/bin/video_processor: /usr/local/cuda-12.6/lib64/libcudart.so
/home/skand/video-processor/bin/video_processor: /usr/local/lib/libopencv_ximgproc.so.4.12.0
/home/skand/video-processor/bin/video_processor: /usr/local/lib/libopencv_imgcodecs.so.4.12.0
/home/skand/video-processor/bin/video_processor: /usr/local/lib/libopencv_cudalegacy.so.4.12.0
/home/skand/video-processor/bin/video_processor: /usr/local/lib/libopencv_cudaimgproc.so.4.12.0
/home/skand/video-processor/bin/video_processor: /usr/local/lib/libopencv_cudafilters.so.4.12.0
/home/skand/video-processor/bin/video_processor: /usr/local/lib/libopencv_cudaarithm.so.4.12.0
/home/skand/video-processor/bin/video_processor: /usr/local/lib/libopencv_objdetect.so.4.12.0
/home/skand/video-processor/bin/video_processor: /usr/local/lib/libopencv_video.so.4.12.0
/home/skand/video-processor/bin/video_processor: /usr/local/lib/libopencv_calib3d.so.4.12.0
/home/skand/video-processor/bin/video_processor: /usr/local/lib/libopencv_features2d.so.4.12.0
/home/skand/video-processor/bin/video_processor: /usr/local/lib/libopencv_flann.so.4.12.0
/home/skand/video-processor/bin/video_processor: /usr/local/lib/libopencv_dnn.so.4.12.0
/home/skand/video-processor/bin/video_processor: /usr/local/lib/libopencv_quality.so.4.12.0
/home/skand/video-processor/bin/video_processor: /usr/local/lib/libopencv_imgproc.so.4.12.0
/home/skand/video-processor/bin/video_processor: /usr/local/lib/libopencv_ml.so.4.12.0
/home/skand/video-processor/bin/video_processor: /usr/local/lib/libopencv_core.so.4.12.0
/home/skand/video-processor/bin/video_processor: /usr/local/lib/libopencv_cudev.so.4.12.0
/home/skand/video-processor/bin/video_processor: /usr/lib/x86_64-linux-gnu/librt.a
/home/skand/video-processor/bin/video_processor: CMakeFiles/video_processor.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/skand/video-processor/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Linking CXX executable /home/skand/video-processor/bin/video_processor"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/video_processor.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/video_processor.dir/build: /home/skand/video-processor/bin/video_processor
.PHONY : CMakeFiles/video_processor.dir/build

CMakeFiles/video_processor.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/video_processor.dir/cmake_clean.cmake
.PHONY : CMakeFiles/video_processor.dir/clean

CMakeFiles/video_processor.dir/depend:
	cd /home/skand/video-processor/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/skand/video-processor /home/skand/video-processor /home/skand/video-processor/build /home/skand/video-processor/build /home/skand/video-processor/build/CMakeFiles/video_processor.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/video_processor.dir/depend

