#!/bin/bash

# Configuration
GPU_ARCH="sm_89"
CUDA_PATH="/usr/local/cuda"
CLANG="clang++-20"
PLUGIN="./build/UvmTrackingPass.so"
ADDITIONAL_FLAGS="-fgpu-rdc -g -O2"
INCLUDE_DIR="-I./include"
LIB_SRC="./libMarkAccess.cu"
TARGET_FILE="../examples/benchmark_kernel_single.cu"

# Check if a filename was provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <source_file.cu> [--run]"
    exit 1
fi

SOURCE_INPUT=$1
RUN_BENCHMARK=false

# Check for run flag
if [[ "$*" == *"--run"* ]]; then
    RUN_BENCHMARK=true
fi

# 1. Copy the provided file to the target location
# cp "$SOURCE_INPUT" "$TARGET_FILE"

# Extract base name for executable naming
FILENAME=$(basename -- "$SOURCE_INPUT")
FILENAME_NO_EXT="${FILENAME%.*}"

EXE_NORMAL="build/${FILENAME_NO_EXT}Normal"
EXE_INSTRUMENTED="build/${FILENAME_NO_EXT}Instrumented"

echo "--- Compiling $FILENAME ---"

# 2. Compile Instrumented Version
# Includes: -DTRACKING_ENABLED, the compiler pass, and the helper library
$CLANG -x cuda --cuda-gpu-arch=$GPU_ARCH \
    $ADDITIONAL_FLAGS \
    $INCLUDE_DIR\
    -DTRACKING_ENABLED \
    -fpass-plugin=$PLUGIN \
    "$SOURCE_INPUT" "$LIB_SRC" \
    --cuda-path=$CUDA_PATH -L$CUDA_PATH/lib64 \
    -lcudart -o "$EXE_INSTRUMENTED"

if [ $? -eq 0 ]; then
    echo "Successfully built: $EXE_INSTRUMENTED"
else
    echo "Error building instrumented version"
    exit 1
fi

# 3. Compile Normal Version
# Excludes: compiler pass and libMarkAccess
$CLANG -x cuda --cuda-gpu-arch=$GPU_ARCH \
    $ADDITIONAL_FLAGS \
    $INCLUDE_DIR \
    "$SOURCE_INPUT" \
    --cuda-path=$CUDA_PATH -L$CUDA_PATH/lib64 \
    -lcudart -o "$EXE_NORMAL"

if [ $? -eq 0 ]; then
    echo "Successfully built: $EXE_NORMAL"
else
    echo "Error building normal version"
    exit 1
fi

# 4. Runtime Benchmark
if [ "$RUN_BENCHMARK" = true ]; then
    echo -e "\n--- Running Benchmarks ---"

    # Function to get time in seconds using 'time'
    # We use 'format %e' to get real elapsed time
    TIME_CMD="time -f %e"

    echo "Running Normal..."
    TIME_NORMAL=$($TIME_CMD ./$EXE_NORMAL 2>&1 >/dev/null)
    
    echo "Running Instrumented..."
    TIME_INST=$($TIME_CMD ./$EXE_INSTRUMENTED 2>&1 >/dev/null)

    # Calculate ratio using bc
    RATIO=$(echo "scale=4; $TIME_INST / $TIME_NORMAL" | bc)

    echo "--------------------------"
    echo "Normal Time:       ${TIME_NORMAL}s"
    echo "Instrumented Time: ${TIME_INST}s"
    echo "Overhead Ratio:    ${RATIO}x"
    echo "--------------------------"
fi