# UVM Access Tracking Testing Framework

This document explains the testing framework used in this repository and how to write a new test.

## Overview

The testing framework is driven by the `run_tests.py` script located in the root of the repository. It is a lightweight, directory-based test runner that automatically discovers and executes tests.

### Test Discovery

The `run_tests.py` script looks for tests in the `tests/` directory. Specifically, it considers any subdirectory inside `tests/` to be a test suite if it contains a `test.py` file. 

For example, the directory structure looks like this:
```
tests/
  ├── control_thread/
  │   ├── test.py
  │   └── long_running_test.cu
  ├── thread_modes/
  │   ├── test.py
  │   └── thread_mode_tests.cu
  └── ...
```

### Execution

When you run `python3 run_tests.py`:
1. It iterates through all discovered test directories.
2. For each directory, it executes its `test.py` file as a separate Python subprocess.
3. The `test.py` scripts are run with the repository root directory (`ROOT_DIR`) as their current working directory (`cwd`). This means all relative paths in your `test.py` scripts should be relative to the repository root, **not** the directory the test resides in.
4. A test is considered **passed** if the `test.py` script exits with a status code of `0`. Any non-zero exit code indicates a **failure**.
5. Standard output and standard error from the test scripts are suppressed unless the test fails or the `-v` / `--verbose` flag is passed to `run_tests.py`.

### Filtering Tests

You can filter which tests to run by passing a substring to `run_tests.py`. It will only run tests whose directory name contains the provided substring (case-insensitive).
```bash
# Run only the thread_modes test
python3 run_tests.py thread_modes
```

## How to Write a Test

To add a new test, follow these steps:

### 1. Create a Test Directory
Create a new subdirectory inside `tests/`. Give it a descriptive name for the feature or component being tested.
```bash
mkdir tests/my_new_feature
```

### 2. Add Test Files
Place your test assets (such as `.cu` or `.c` files) inside this new directory.
```bash
touch tests/my_new_feature/my_test_kernel.cu
```

### 3. Write `test.py`
Create a `test.py` file inside your test directory. This script will coordinate the compilation, execution, and validation of your test.
```bash
touch tests/my_new_feature/test.py
```

### 4. Implement the Test Logic in `test.py`
Your `test.py` script should typically perform the following steps:
1. **Compile**: Compile the CUDA code or binary you need for the test. Remember that the working directory is the repository root, so use paths relative to the root (e.g., `tests/my_new_feature/my_test_kernel.cu`).
2. **Execute**: Launch the compiled binary. You may need to set specific environment variables like `LD_PRELOAD`.
3. **Verify**: Parse the output, log files, or communicate with the process (e.g., over UNIX domain sockets) to verify that the behavior matches expectations.
4. **Exit cleanly**: 
   - Exit with code `0` (`sys.exit(0)`) if everything is correct.
   - Exit with a non-zero code (e.g., `sys.exit(1)`) or raise an unhandled exception if the test fails.

#### Example `test.py` Skeleton
```python
import subprocess
import sys
import os

# Define paths relative to the repository root
SOURCE_FILE = "tests/my_new_feature/my_test_kernel.cu"
EXECUTABLE = "./build/my_test_kernel"

def compile_kernel():
    # Compile the test binary
    cmd = ["nvcc", SOURCE_FILE, "-o", EXECUTABLE]
    subprocess.run(cmd, check=True)

def run_and_verify():
    # Run the test binary
    proc = subprocess.run([EXECUTABLE], capture_output=True, text=True)
    
    # Check for success
    if proc.returncode != 0:
        print(f"Test executable failed with return code {proc.returncode}")
        print(proc.stderr)
        sys.exit(1)
        
    if "SUCCESS" not in proc.stdout:
        print("Test failed: 'SUCCESS' not found in output.")
        sys.exit(1)
        
    print("Test passed.")

if __name__ == "__main__":
    try:
        compile_kernel()
        run_and_verify()
    except Exception as e:
        print(f"Test script failed: {e}")
        sys.exit(1)
```

## Running Tests
Run the entire test suite from the repository root:
```bash
python3 run_tests.py
```
For verbose output:
```bash
python3 run_tests.py -v
```
