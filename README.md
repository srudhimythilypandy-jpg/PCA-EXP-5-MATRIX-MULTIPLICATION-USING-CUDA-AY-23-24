# PCA-EXP-5-MATRIX-MULTIPLICATION-USING-CUDA-AY-23-24
<h3>AIM:</h3>
<h3>ENTER YOUR NAME</h3>
<h3>ENTER YOUR REGISTER NO</h3>
<h3>EX. NO</h3>
<h3>DATE</h3>
<h1> <align=center> MATRIX MULTIPLICATION USING CUDA </h3>
  Implement Matrix Multiplication using GPU.</h3>

## AIM:
To perform Matrix Multiplication using CUDA and check its performance with nvprof.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
## PROCEDURE:
1.	Define Constants: Define the size of the matrices (SIZE) and the size of the CUDA blocks (BLOCK_SIZE).
2.	Kernel Function: Define a CUDA kernel function matrixMultiply that performs the matrix multiplication.
3.	In the main function, perform the following steps:
4.	Initialize Matrices: Initialize the input matrices ‘a’ and ‘b’ with some values.
5.	Allocate Device Memory: Allocate memory on the GPU for the input matrices ‘a’ and ‘b’, and the output matrix ‘c’.
6.	Copy Matrices to Device: Copy the input matrices from host (CPU) memory to device (GPU) memory.
7.	Set Grid and Block Sizes: Set the grid and block sizes for the CUDA kernel launch.
8.	Start Timer: Start a timer to measure the execution time of the kernel.
9.	Launch Kernel: Launch the matrixMultiply kernel with the appropriate grid and block sizes, and the input and output matrices as arguments.
10.	Copy Result to Host: After the kernel execution, copy the result matrix from device memory to host memory.
11.	Stop Timer: Stop the timer and calculate the elapsed time.
12.	Print Result: Print the result matrix and the elapsed time.
13.	Free Device Memory: Finally, free the device memory that was allocated for the matrices.
## PROGRAM:
%%writefile matmul.cu #include <stdio.h> #include <cuda_runtime.h> #include <sys/time.h>

#define SIZE 4 #define BLOCK_SIZE 2

// Kernel function to perform matrix multiplication global void matrixMultiply(int *a, int *b, int *c, int size) { int row = blockIdx.y * blockDim.y + threadIdx.y; int col = blockIdx.x * blockDim.x + threadIdx.x;

// Check if the thread is within the matrix bounds
if (row < size && col < size) {
    int sum = 0;
    for (int k = 0; k < size; ++k)
    {
        sum += a[row * size + k] * b[k * size + col];
    }
    c[row * size + col] = sum;
}

// Helper function for matrix multiplication on CPU (for verification) void matrixMultiplyCPU(int a[][SIZE], int b[][SIZE], int c[][SIZE], int size) { for (int i = 0; i < size; ++i) { for (int j = 0; j < size; ++j) { c[i][j] = 0; for (int k = 0; k < size; ++k) { c[i][j] += a[i][k] * b[k][j]; } } } }

int main() { int a[SIZE][SIZE], b[SIZE][SIZE], c[SIZE][SIZE] = {0}; int cpu_c[SIZE][SIZE] = {0}; // Results from CPU calculation for verification int *dev_a, *dev_b, *dev_c; int size = SIZE * SIZE * sizeof(int);

// Initialize matrices 'a' and 'b'
for (int i = 0; i < SIZE; ++i)
{
    for (int j = 0; j < SIZE; ++j)
    {
        a[i][j] = i + j;
        b[i][j] = i - j;
    }
}

// Print input matrices
printf("Matrix A:\n");
for (int i = 0; i < SIZE; ++i) {
    for (int j = 0; j < SIZE; ++j) {
        printf("%d ", a[i][j]);
    }
    printf("\n");
}

printf("Matrix B:\n");
for (int i = 0; i < SIZE; ++i) {
    for (int j = 0; j < SIZE; ++j) {
        printf("%d ", b[i][j]);
    }
    printf("\n");
}

// Compute on CPU for verification
printf("Computing matrix multiplication on CPU for verification...\n");
matrixMultiplyCPU(a, b, cpu_c, SIZE);

printf("CPU Result Matrix:\n");
for (int i = 0; i < SIZE; ++i) {
    for (int j = 0; j < SIZE; ++j) {
        printf("%d ", cpu_c[i][j]);
    }
    printf("\n");
}

// Check if CUDA is available
int deviceCount = 0;
cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

if (error_id != cudaSuccess) {
    printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
    printf("Result: FAILED\n");
    return 1;
}

// This function call returns 0 if there are no CUDA capable devices.
if (deviceCount == 0) {
    printf("There are no available device(s) that support CUDA\n");
    printf("Using CPU result as final\n");
    return 0;
} else {
    printf("Detected %d CUDA Capable device(s)\n", deviceCount);
}

// Allocate memory on the device
cudaError_t err;

err = cudaMalloc((void**)&dev_a, size);
if (err != cudaSuccess) {
    printf("CUDA error in cudaMalloc: %s\n", cudaGetErrorString(err));
    return 1;
}

err = cudaMalloc((void**)&dev_b, size);
if (err != cudaSuccess) {
    printf("CUDA error in cudaMalloc: %s\n", cudaGetErrorString(err));
    cudaFree(dev_a);
    return 1;
}

err = cudaMalloc((void**)&dev_c, size);
if (err != cudaSuccess) {
    printf("CUDA error in cudaMalloc: %s\n", cudaGetErrorString(err));
    cudaFree(dev_a);
    cudaFree(dev_b);
    return 1;
}

// Copy input matrices from host to device memory
err = cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
if (err != cudaSuccess) {
    printf("CUDA error in cudaMemcpy: %s\n", cudaGetErrorString(err));
    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);
    return 1;
}

err = cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
if (err != cudaSuccess) {
    printf("CUDA error in cudaMemcpy: %s\n", cudaGetErrorString(err));
    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);
    return 1;
}

// Initialize dev_c with zeros
err = cudaMemset(dev_c, 0, size);
if (err != cudaSuccess) {
    printf("CUDA error in cudaMemset: %s\n", cudaGetErrorString(err));
    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);
    return 1;
}

// Set grid and block sizes
dim3 dimGrid((SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, (SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

// Start timer
struct timeval start, end;
gettimeofday(&start, NULL);

// Launch kernel
printf("Launching CUDA kernel...\n");
matrixMultiply<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, SIZE);

// Check for kernel launch errors
err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("CUDA error in kernel launch: %s\n", cudaGetErrorString(err));
    printf("Using CPU result as final\n");

    // Copy CPU result to c
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            c[i][j] = cpu_c[i][j];
        }
    }

    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);

    // Print the CPU result as the final result
    printf("Final Result Matrix (from CPU):\n");
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            printf("%d ", c[i][j]);
        }
        printf("\n");
    }

    gettimeofday(&end, NULL);
    double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
    printf("Elapsed Time (CPU): %.6f seconds\n", elapsed_time);

    return 0;
}

// Wait for GPU to finish
err = cudaDeviceSynchronize();
if (err != cudaSuccess) {
    printf("CUDA error in synchronize: %s\n", cudaGetErrorString(err));
    printf("Using CPU result as final\n");

    // Copy CPU result to c
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            c[i][j] = cpu_c[i][j];
        }
    }

    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);

    // Print the CPU result as the final result
    printf("Final Result Matrix (from CPU):\n");
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            printf("%d ", c[i][j]);
        }
        printf("\n");
    }

    gettimeofday(&end, NULL);
    double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
    printf("Elapsed Time (CPU): %.6f seconds\n", elapsed_time);

    return 0;
}

// Copy result matrix from device to host memory
err = cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);
if (err != cudaSuccess) {
    printf("CUDA error in cudaMemcpy: %s\n", cudaGetErrorString(err));
    printf("Using CPU result as final\n");

    // Copy CPU result to c
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            c[i][j] = cpu_c[i][j];
        }
    }

    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);

    // Print the CPU result as the final result
    printf("Final Result Matrix (from CPU):\n");
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            printf("%d ", c[i][j]);
        }
        printf("\n");
    }

    gettimeofday(&end, NULL);
    double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
    printf("Elapsed Time (CPU): %.6f seconds\n", elapsed_time);

    return 0;
}

// Stop timer
gettimeofday(&end, NULL);
double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;

// Print the result matrix
printf("GPU Result Matrix:\n");
for (int i = 0; i < SIZE; ++i) {
    for (int j = 0; j < SIZE; ++j) {
        printf("%d ", c[i][j]);
    }
    printf("\n");
}

// Print the elapsed time
printf("Elapsed Time (GPU): %.6f seconds\n", elapsed_time);

// Free device memory
cudaFree(dev_a);
cudaFree(dev_b);
cudaFree(dev_c);

return 0;

%%bash nvcc -c matmul.cu -o matmul.o -ccbin /usr/bin/g++ g++ matmul.o -o matmul -L/usr/local/cuda/lib64 -lcudart -lcuda -lrt -lpthread ./matmul

## OUTPUT:
<img width="758" height="645" alt="Screenshot 2026-09-01 104207" src="https://github.com/user-attachments/assets/92f7f871-f8bf-4130-8ec3-c5d9a851a87e" />


## RESULT:
Thus the program has been executed by using CUDA to mulptiply two matrices. It is observed that there are variations in host and device elapsed time. Device took 0.126788 seconds time and host took 0.000294 time.
