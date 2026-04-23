FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Avoid interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Basic tools + OpenSSH Server
RUN apt update && apt install -y \
    curl \
    cmake \
    lsb-release \
    software-properties-common \
    gnupg \
    libstdc++-12-dev \
    libc6-dev \
    openssh-server

# Add LLVM 20 repo and install
RUN curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /usr/share/keyrings/llvm-archive-keyring.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/llvm-archive-keyring.gpg] http://apt.llvm.org/jammy/ llvm-toolchain-jammy-20 main" | tee /etc/apt/sources.list.d/llvm.list
RUN apt update && apt install -y \
    clang-20 \
    llvm-20

# --- SSH Setup ---

# 1. Create run directory for SSH
RUN mkdir /var/run/sshd

# 2. Set root password (change 'password123' to something else if you prefer)
RUN echo 'root:password123' | chpasswd

# 3. Allow Root Login and Password Authentication
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

# 4. Standard SSH port
EXPOSE 22

# Set working directory
WORKDIR /data

# Start SSH service and then keep the container alive with sleep infinity
CMD ["sh", "-c", "service ssh start && sleep infinity"]