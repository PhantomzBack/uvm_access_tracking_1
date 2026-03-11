cmake_minimum_required(VERSION 3.15)
project(UvmTracker)

# 1. Find the pre-installed LLVM
find_package(LLVM REQUIRED CONFIG)

# 2. Add LLVM include directories and definitions
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# 3. Create the plugin (The .so file)
add_library(UvmTrackingPass MODULE UvmTrackingPass.cpp)

# 4. Critical: Ensure the plugin is compatible with the New Pass Manager
set_target_properties(UvmTrackingPass PROPERTIES
    COMPILE_FLAGS "-fno-rtti"
    PREFIX ""
    SUFFIX ".so"
)

target_link_libraries(UvmTrackingPass LLVM)
