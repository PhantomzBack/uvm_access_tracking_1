FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Avoid interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Basic tools
RUN apt update && apt install -y \
    curl \
    cmake \
    lsb-release \
    software-properties-common \
    gnupg \
    libstdc++-12-dev \
    libc6-dev

# Add LLVM 20 repo and install
RUN curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /usr/share/keyrings/llvm-archive-keyring.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/llvm-archive-keyring.gpg] http://apt.llvm.org/jammy/ llvm-toolchain-jammy-20 main" | tee /etc/apt/sources.list.d/llvm.list
RUN apt update && apt install -y \
    clang-20 \
    llvm-20

# Set working directory
WORKDIR /data
