#include <iostream>
#include <stdlib.h>
#include <math.h>

using namespace std;

#include <cuda_runtime.h>

// Variables
int elements1D;
int width;
int numOfElements;
size_t datasize;

// Data pointers
float *A_seq;
float *B_seq;
float *C_seq;
float *C;

// CUDA pointers to device buffers
float *d_A;
float *d_B;
float *d_C;

// CUDA events
cudaEvent_t startWriteEvent;
cudaEvent_t stopWriteEvent;
cudaEvent_t startKernelEvent;
cudaEvent_t stopKernelEvent;
cudaEvent_t startReadEvent;
cudaEvent_t stopReadEvent;

// Time variables
float kernelTime;
float writeTime;
float readTime;

__global__ void matrix_multiplication(float *a, float *b, float *c, int sizeX, int sizeY, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < sizeX && j < sizeY) {
        float sum = 0.0f;
        for (int k = 0; k < width; k++) {
            sum += a[k * sizeX + i] * b[j * width + k];
        }
        c[j * sizeX + i] = sum;
    }
}

float getTimeInMilliseconds(cudaEvent_t start_event, cudaEvent_t stop_event) {
    cudaEventSynchronize(stop_event);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);
    return milliseconds;
}

void initializeCUDAEvents() {
    cudaEventCreate(&startWriteEvent);
    cudaEventCreate(&stopWriteEvent);
    cudaEventCreate(&startKernelEvent);
    cudaEventCreate(&stopKernelEvent);
    cudaEventCreate(&startReadEvent);
    cudaEventCreate(&stopReadEvent);
}

void initializeInputDataOnHostAndPlatform() {
    A_seq = (float *) malloc(datasize);
    B_seq = (float *) malloc(datasize);
    C_seq = (float *) malloc(datasize);
    C = (float *) malloc(datasize);

    for (int i = 0; i < numOfElements; i++) {
        A_seq[i] = (float)rand() / (float)RAND_MAX;
        B_seq[i] = (float)rand() / (float)RAND_MAX;
        C_seq[i] = -1;
        C[i] = -1;
    }
}

void createDataBuffers() {
    cudaMalloc(&d_A, datasize);
    cudaMalloc(&d_B, datasize);
    cudaMalloc(&d_C, datasize);
}

void transferInputDataToPlatform() {
    cudaError_t status;
    status = cudaMemcpy(d_A, A_seq, datasize, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        cout << "Error in cudaMemcpy for array d_A, status: " << status << endl;
    }
    status = cudaMemcpy(d_B, B_seq, datasize, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        cout << "Error in cudaMemcpy for array d_B, status: " << status << endl;
    }
}

void launchKernel() {
    dim3 grimDim(elements1D, elements1D);
    dim3 blockDimPerGrid(1, 1);
    if (numOfElements > elements1D){
        blockDimPerGrid.x = ceil(double(elements1D)/double(grimDim.x));
        blockDimPerGrid.y = ceil(double(elements1D)/double(grimDim.y));
    }

    matrix_multiplication<<<grimDim,blockDimPerGrid>>>(d_A, d_B, d_C, elements1D, elements1D, width);
    cudaDeviceSynchronize();
}

void transferResultDataFromPlatform() {
    cudaError_t status = cudaMemcpy(C, d_C, datasize, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        cout << "Error in cudaMemcpy for array d_C, status: " << status << endl;
    }
    cudaDeviceSynchronize();
}

void validateResultsOfKernel(float* C, float* C_seq) {
    bool valid = true;
    for (int i = 0; i < numOfElements; i++) {
        float diff = fabs(C[i] - C_seq[i]);
        if (diff > 0.1f) {
            cout << "C_seq[" << i << "]: " << C_seq[i] << " - C[" << i << "]: " << C[i] << endl;
            valid = false;
            break;
        }
    }

    if (valid) {
        cout << "Result is correct" << endl;
    } else {
        cout << "Result is not correct" << endl;
    }
    cout << "\n";
}

void freeMemory() {
    cudaEventDestroy(startWriteEvent);
    cudaEventDestroy(stopWriteEvent);
    cudaEventDestroy(startKernelEvent);
    cudaEventDestroy(stopKernelEvent);
    cudaEventDestroy(startReadEvent);
    cudaEventDestroy(stopReadEvent);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A_seq);
    free(B_seq);
    free(C_seq);
    free(C);
}

void matrixMultiplication(float* A_seq, float* B_seq, float* C_seq, int sizeX, int sizeY, int width) {
    for (int i = 0; i < sizeX; i++) {
        for (int j = 0; j < sizeY; j++) {
            float sum = 0.0f;
            for (int k = 0; k < width; k++) {
                sum += A_seq[k * sizeX + i] * B_seq[j*width + k];
            }
            C_seq[j * sizeX + i] = sum;
        }
    }
}

int main(int argc, char **argv) {
    if (argc > 1) {
        elements1D = atoi(argv[1]);
        width = elements1D; // Square matrices (MxM)
        numOfElements = elements1D * elements1D;
        datasize = sizeof(float) * numOfElements;
    } else {
        cout << "Run: ./host <elements>" << endl;
        return -1;
    }

    cout << "CUDA Matrix Multiplication" << endl;
    cout << "Number of Elements: " << numOfElements << endl;

    initializeCUDAEvents();
    createDataBuffers();

    // Initialization of input data both on the host and on the area mapped to the platform buffers
    initializeInputDataOnHostAndPlatform();

    // Transfer input data (Section 3.2.2)
    cudaEventRecord(startWriteEvent);
    transferInputDataToPlatform();
    cudaEventRecord(stopWriteEvent);

    writeTime = getTimeInMilliseconds(startWriteEvent, stopWriteEvent);

    // Launch kernel (Section 3.2.3)
    cudaEventRecord(startKernelEvent);
    launchKernel();
    cudaEventRecord(stopKernelEvent);
    kernelTime = getTimeInMilliseconds(startKernelEvent, stopKernelEvent);

    // Transfer result data (Section 3.2.4)
    cudaEventRecord(startReadEvent);
    transferResultDataFromPlatform();
    cudaEventRecord(stopReadEvent);
    readTime = getTimeInMilliseconds(startReadEvent, stopReadEvent);

    // Report the timing from OpenCL events
    cout << "\nTransferring Input Data Time   : " << writeTime << " milliseconds." << endl;
    cout << "Kernel Execution Time          : " << kernelTime << " milliseconds." << endl;
    cout << "Transferring Result Data Time  : " << readTime << " milliseconds." << endl;
    cout << "\n";

    matrixMultiplication(A_seq, B_seq, C_seq, elements1D, elements1D, width);

    // Validate the results of the kernel
    validateResultsOfKernel(C, C_seq);

    freeMemory();

    return 0;
}
