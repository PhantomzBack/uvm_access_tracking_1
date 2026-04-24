#!/bin/bash

# Configuration
GPU_ARCH="sm_89"
CUDA_PATH="/usr/local/cuda"
CLANG="clang++-20"
PLUGIN="./build/UvmTrackingPass.so"
ADDITIONAL_FLAGS="-fgpu-rdc -g -O2"
INCLUDE_DIR="-I./include -I../cutlass/include"
LIB_SRC="./libMarkAccess.cu"
TARGET_FILE="../examples/benchmark_kernel_single.cu"
OTHER_LIBRARIES="-lcudart -lcublas"

# Check if a filename was provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <source_file.cu> [--run] [--preload] [--mode {skip|alloc|none}]"
    exit 1
fi

SOURCE_INPUT=$1
RUN_BENCHMARK=false
USE_PRELOAD=false
MODE=""

# Parse flags
for arg in "$@"; do
    if [ "$arg" == "--run" ]; then
        RUN_BENCHMARK=true
    fi
    if [ "$arg" == "--preload" ]; then
        USE_PRELOAD=true
    fi
    if [ "$arg" == "--mode" ]; then
        MODE="next"   # marker: next arg is the mode value
    elif [ "$MODE" == "next" ]; then
        MODE="$arg"
    fi
done

# Validate mode and set default
if [ -z "$MODE" ]; then
    if [ "$USE_PRELOAD" = true ]; then
        MODE="skip"
    else
        MODE="alloc"
    fi
fi

case "$MODE" in
    skip|alloc|none)
        MODE_FLAG="-DTRACKING_MODE_${MODE^^}"
        ;;
    *)
        echo "Error: invalid mode '$MODE'. Use skip, alloc, or none."
        exit 1
        ;;
esac

echo "--- Mode: $MODE ($MODE_FLAG) ---"

# Preload shared library
PRELOAD_SO="./libMallocIntercept.so"
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
    # -rdynamic makes executable symbols visible to the LD_PRELOAD wrapper
    ADDITIONAL_FLAGS="$ADDITIONAL_FLAGS -rdynamic"
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
# Includes: -DTRACKING_ENABLED, mode flag, the compiler pass, and the helper library
$CLANG -x cuda --cuda-gpu-arch=$GPU_ARCH \
    $ADDITIONAL_FLAGS \
    $INCLUDE_DIR \
    -DTRACKING_ENABLED \
    $MODE_FLAG \
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

    # Export LD_PRELOAD if needed (applies to all commands in this section)
    if [ "$USE_PRELOAD" = true ]; then
        export LD_PRELOAD="$PRELOAD_SO"
    fi

    TIME_CMD="/usr/bin/time -f %e"

    echo "Running Normal..."
    TIME_FILE=$(mktemp)
    $TIME_CMD -o "$TIME_FILE" ./$EXE_NORMAL >/dev/null 2>&1
    EXIT_NORMAL=$?
    TIME_NORMAL=$(cat "$TIME_FILE")
    rm -f "$TIME_FILE"

    if [ $EXIT_NORMAL -ne 0 ]; then
        echo "Warning: Normal binary exited with status $EXIT_NORMAL"
    fi

    echo "Running Instrumented..."
    TIME_FILE=$(mktemp)
    $TIME_CMD -o "$TIME_FILE" ./$EXE_INSTRUMENTED >/dev/null 2>&1
    EXIT_INST=$?
    TIME_INST=$(cat "$TIME_FILE")
    rm -f "$TIME_FILE"

    if [ $EXIT_INST -ne 0 ]; then
        echo "Warning: Instrumented binary exited with status $EXIT_INST"
    fi

    # Calculate ratio using awk (bc may not be installed)
    RATIO=$(awk "BEGIN {printf \"%.4f\", $TIME_INST / $TIME_NORMAL}")

    echo "--------------------------"
    echo "Normal Time:       ${TIME_NORMAL}s"
    echo "Instrumented Time: ${TIME_INST}s"
    echo "Overhead Ratio:    ${RATIO}x"
    echo "--------------------------"
fi
