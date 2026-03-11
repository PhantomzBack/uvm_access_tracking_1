#include <stdio.h>
// Exampl
__global__ void myKernel() {
    printf("Hello from GPU thread %d\\n", threadIdx.x);
}

int main() {
    myKernel<<<1, 5>>>();
    // Add synchronization point
    cudaError_t cudaerr = cudaDeviceSynchronize(); 
    if (cudaerr != cudaSuccess) {
        printf("Kernel launch failed with error: %s\\n", cudaGetErrorString(cudaerr));
    }
    return 0;
}

