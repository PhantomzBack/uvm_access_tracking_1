#!/bin/bash

# Configuration
CUDA_PATH="/usr/local/cuda"
CLANG="clang++-20"
PLUGIN="./build/UvmTrackingPass.so"
ADDITIONAL_FLAGS="-fgpu-rdc -g -O2"
INCLUDE_DIR="-I./include -I../cutlass/include"
LIB_SRC="./libMarkAccess.cu"
TARGET_FILE="../examples/benchmark_kernel_single.cu"
OTHER_LIBRARIES="-lcudart -lcublas"

# Detect GPU architecture dynamically
GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
GPU_ARCH="sm_${GPU_ARCH}"

# Check if a filename was provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <source_file.cu> [--run] [--preload] [--mode no-preload|preload-alloc|preload-only]"
    exit 1
fi

SOURCE_INPUT=$1
shift  # remove source file from remaining args

RUN_BENCHMARK=false
USE_PRELOAD=false
MODE="no-preload"

# Parse remaining flags
for arg in "$@"; do
    if [ "$arg" == "--run" ]; then
        RUN_BENCHMARK=true
    elif [ "$arg" == "--preload" ]; then
        USE_PRELOAD=true
    elif [ "$arg" == "--mode" ]; then
        :  # value handled below
    fi
done

# Handle --mode <value>
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
    if [ "${ARGS[$i]}" == "--mode" ]; then
        j=$((i+1))
        if [ $j -lt ${#ARGS[@]} ]; then
            MODE="${ARGS[$j]}"
        fi
    fi
done

# Map mode to compile flag and preload requirement
case "$MODE" in
    no-preload)
        MODE_FLAG=""
        ;;
    preload-alloc)
        MODE_FLAG="-DUVM_TRACKING_MODE=1"
        USE_PRELOAD=true
        ;;
    preload-only)
        MODE_FLAG="-DUVM_TRACKING_MODE=2"
        USE_PRELOAD=true
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Valid modes: no-preload, preload-alloc, preload-only"
        exit 1
        ;;
esac

# Preload shared library
PRELOAD_SO="./libMallocIntercept.so"
LD_PRELOAD_CMD=""
if [ "$USE_PRELOAD" = true ]; then
    echo "--- Preload mode enabled ---"
    if [ ! -f "$PRELOAD_SO" ]; then
        echo "Building $PRELOAD_SO ..."
        clang++-20 -shared -fPIC -O2 -I"$CUDA_PATH/include" libMallocIntercept.cpp -o "$PRELOAD_SO" -ldl
        if [ $? -ne 0 ]; then
            echo "Error building $PRELOAD_SO"
            exit 1
        fi
    fi
    LD_PRELOAD_CMD="LD_PRELOAD=$PRELOAD_SO"
    # -rdynamic makes executable symbols visible to the LD_PRELOAD wrapper
    ADDITIONAL_FLAGS="$ADDITIONAL_FLAGS -rdynamic"
fi

# Extract base name for executable naming
FILENAME=$(basename -- "$SOURCE_INPUT")
FILENAME_NO_EXT="${FILENAME%.*}"

EXE_NORMAL="build/${FILENAME_NO_EXT}Normal"
EXE_INSTRUMENTED="build/${FILENAME_NO_EXT}Instrumented"
if [ "$MODE" != "no-preload" ]; then
    EXE_INSTRUMENTED="build/${FILENAME_NO_EXT}Instrumented_${MODE}"
fi

echo "--- Compiling $FILENAME (mode: $MODE) ---"

# 2. Compile Instrumented Version
# Includes: -DTRACKING_ENABLED, the compiler pass, and the helper library
$CLANG -x cuda --cuda-gpu-arch=$GPU_ARCH \
    $ADDITIONAL_FLAGS \
    $INCLUDE_DIR \
    $MODE_FLAG \
    -DTRACKING_ENABLED \
    -fpass-plugin=$PLUGIN \
    "$SOURCE_INPUT" "$LIB_SRC" \
    --cuda-path=$CUDA_PATH -L$CUDA_PATH/lib64 \
    -lcudart -lcublas -o "$EXE_INSTRUMENTED"

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
    -lcudart -lcublas -o "$EXE_NORMAL"

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
    TIME_CMD="/usr/bin/time -f %e"

    echo "Running Normal..."
    TIME_NORMAL=$(
        if [ "$USE_PRELOAD" = true ]; then
            export LD_PRELOAD="$PRELOAD_SO"
        fi
        $TIME_CMD ./$EXE_NORMAL 2>&1 >/dev/null
    )
    
    echo "Running Instrumented..."
    TIME_INST=$(
        if [ "$USE_PRELOAD" = true ]; then
            export LD_PRELOAD="$PRELOAD_SO"
        fi
        $TIME_CMD ./$EXE_INSTRUMENTED 2>&1 >/dev/null
    )

    # Calculate ratio using awk (bc may not be installed)
    RATIO=$(awk "BEGIN {printf \"%.4f\", $TIME_INST / $TIME_NORMAL}")

    echo "--------------------------"
    echo "Normal Time:       ${TIME_NORMAL}s"
    echo "Instrumented Time: ${TIME_INST}s"
    echo "Overhead Ratio:    ${RATIO}x"
    echo "--------------------------"
fi
