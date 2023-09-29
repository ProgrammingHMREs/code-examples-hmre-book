#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

using namespace std;

#define CL_TARGET_OPENCL_VERSION 120

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

const int LOCAL_WORK_SIZE = 16;

// Variables
int elements1D;
int width;
int numOfElements;
size_t datasize;
char *source;
char *programSourceFileName = (char*)"kernel.cl";

// Data pointers
float *A;
float *B;
float *C;
float *A_seq;
float *B_seq;
float *C_seq;

// OpenCL variables
string platformName;
cl_uint numPlatforms;
cl_uint numDevices;
cl_platform_id *platforms;
cl_device_id *devices;
cl_context context;
cl_command_queue commandQueue;
cl_kernel kernel;
cl_program program;

// OpenCL buffer types
cl_mem d_A;
cl_mem d_B;
cl_mem d_C;

// OpenCL events
cl_event writeEvent1;
cl_event writeEvent2;
cl_event kernelEvent;
cl_event readEvent;

// Time variables
long kernelTime;
long writeTime;
long readTime;

long getTimeInNanoseconds(cl_event event) {
    clWaitForEvents(1, &event);
    cl_ulong time_start, time_end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    return (time_end - time_start);
}

char *readsource(const char *sourceFilename) {
    FILE *fp;
    int err;
    int size;
    char *source;

    fp = fopen(sourceFilename, "rb");

    if (fp == NULL) {
        printf("Could not open kernel file: %s\n", sourceFilename);
        exit(-1);
    }

    err = fseek(fp, 0, SEEK_END);

    if (err != 0) {
        printf("Error seeking to end of file\n");
        exit(-1);

    }
    size = ftell(fp);

    if (size < 0) {
        printf("Error getting file position\n");
        exit(-1);
    }

    err = fseek(fp, 0, SEEK_SET);
    if (err != 0) {
        printf("Error seeking to start of file\n");
        exit(-1);

    }

    source = (char *) malloc(size + 1);

    if (source == NULL) {
        printf("Error allocating %d bytes for the program source\n", size + 1);
        exit(-1);
    }

    err = fread(source, 1, size, fp);
    if (err != size) {
        printf("only read %d bytes\n", err);
        exit(0);
    }

    source[size] = '\0';
    return source;
}

int discoverPlatformsAndSelect(int platformId) {
    cl_int status;
    cl_uint numPlatforms = 0;

    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (numPlatforms == 0) {
        cout << "No platform detected" << endl;
        return status;
    }

    platforms = (cl_platform_id *) malloc(numPlatforms * sizeof(cl_platform_id));
    if (platforms == NULL) {
        cout << "malloc platform_id failed" << endl;
        return status;
    }

    status = clGetPlatformIDs(numPlatforms, platforms, NULL);
    if (status != CL_SUCCESS) {
        cout << "clGetPlatformIDs failed" << endl;
        return status;
    }

    if (platformId > numPlatforms) {
        cout << "The target platform id is not valid. The number of discovered platforms are: " << numPlatforms << endl;
        return -1;
    }

    cout << numPlatforms << " platforms have been detected" << endl;
    for (int i = 0; i < numPlatforms; i++) {
        char buf[10000];
        cout << "Platform: " << i << endl;
        status = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(buf), buf, NULL);
        if (i == platformId) {
            platformName += buf;
        }
        cout << "\tVendor: " << buf << endl;
        status = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(buf), buf, NULL);
        cout << "\tName  : " << buf << endl;
    }

    cl_platform_id platform = platforms[platformId];
    cout << "Using platform: " << platformId << " --> " << platformName << endl;

    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);

    if (status != CL_SUCCESS) {
        cout << "[WARNING] Using CPU, no GPU available" << endl;
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);
        devices = (cl_device_id *) malloc(numDevices * sizeof(cl_device_id));
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, devices, NULL);
    } else {
        devices = (cl_device_id *) malloc(numDevices * sizeof(cl_device_id));
        cout << "Using accelerator" << endl;
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);

        char buf[1000];
        clGetDeviceInfo(devices[0], CL_DEVICE_NAME, sizeof(buf), buf, NULL);
        cout << "\tDEVICE NAME: " << buf << endl;
    }

    return status;
}

int initializeDataStructures() {
    cl_int status;
    context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);
    if (status != CL_SUCCESS) {
        cout << "Error in clCreateContext" << endl;
        return status;
    }

    commandQueue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &status);
    if (status != CL_SUCCESS || commandQueue == NULL) {
        cout << "Error in clCreateCommandQueue" << endl;
        return status;
    }
    return status;
}

int createKernelFromProgramSource(char* sourceFile) {
    cl_int status;
    // Build from source
    source = readsource(sourceFile);
    program = clCreateProgramWithSource(context, 1, (const char **) &source, NULL, &status);
    if (CL_SUCCESS != status) {
        cout << "Error in clCreateProgramWithSource, status: " << status << endl;
        return status;
    }
    status = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);
    if (CL_SUCCESS != status) {
        cout << "Error in clBuildProgram, status: " << status << endl;
        return status;
    }
    kernel = clCreateKernel(program, "matrixMultiplication", &status);
    if (CL_SUCCESS != status) {
        cout << "Error in clCreateKernel, matrixMultiplication kernel, status: " << status << endl;
        return status;
    }

    return status;
}

void initializeInputDataOnHostAndPlatform() {
    A_seq = (float *) malloc(datasize);
    B_seq = (float *) malloc(datasize);
    C_seq = (float *) malloc(datasize);

    for (int i = 0; i < numOfElements; i++) {
        A_seq[i] = (float)rand() / (float)RAND_MAX;
        B_seq[i] = (float)rand() / (float)RAND_MAX;
        C_seq[i] = -1;
        A[i] = A_seq[i];
        B[i] = B_seq[i];
        C[i] = C_seq[i];
    }
}

int createDataBuffers() {
    cl_int status;
    d_A = clCreateBuffer(context, CL_MEM_READ_WRITE, datasize, NULL, &status);
    if (CL_SUCCESS != status) {
        cout << "Error in clCreateBuffer for array d_A, status: " << status << endl;
    }
    d_B = clCreateBuffer(context, CL_MEM_READ_WRITE, datasize, NULL, &status);
    if (CL_SUCCESS != status) {
        cout << "Error in clCreateBuffer for array d_B, status: " << status << endl;
    }
    d_C = clCreateBuffer(context, CL_MEM_READ_WRITE, datasize, NULL, &status);
    if (CL_SUCCESS != status) {
        cout << "Error in clCreateBuffer for array d_C, status: " << status << endl;
    }

    A = (float *) clEnqueueMapBuffer(commandQueue, d_A, CL_TRUE, CL_MAP_WRITE, 0, datasize, 0, NULL, NULL, NULL);
    B = (float *) clEnqueueMapBuffer(commandQueue, d_B, CL_TRUE, CL_MAP_WRITE, 0, datasize, 0, NULL, NULL, NULL);
    C = (float *) clEnqueueMapBuffer(commandQueue, d_C, CL_TRUE, CL_MAP_READ, 0, datasize, 0, NULL, NULL, NULL);

    return status;
}

void transferInputDataToPlatform() {
    clEnqueueWriteBuffer(commandQueue, d_A, CL_TRUE, 0, datasize, A, 0, NULL, &writeEvent1);
    clEnqueueWriteBuffer(commandQueue, d_B, CL_TRUE, 0, datasize, B, 0, NULL, &writeEvent2);
    clFlush(commandQueue);
}

int setKernelArgs() {
    cl_int status;
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    status |= clSetKernelArg(kernel, 3, sizeof(cl_int), &elements1D);
    status |= clSetKernelArg(kernel, 4, sizeof(cl_int), &elements1D);
    status |= clSetKernelArg(kernel, 5, sizeof(cl_int), &elements1D);

    if (CL_SUCCESS != status) {
        cout << "Error in clSetKernelArg, status: " << status << endl;
    }
    return status;
}

int launchKernel() {
    size_t globalWorkSize[2];
    size_t localWorkSize[3];

    globalWorkSize[0] = elements1D;
    globalWorkSize[1] = elements1D;
    localWorkSize[0] = LOCAL_WORK_SIZE;
    localWorkSize[1] = 1;
    localWorkSize[2] = 1;

    return clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &kernelEvent);
}

void transferResultDataFromPlatform() {
    clEnqueueReadBuffer(commandQueue, d_C, CL_TRUE, 0, datasize, C, 0, NULL, &readEvent);
    clFlush(commandQueue);
}

void validateResultsOfKernel(float* C, float* C_seq) {
    bool valid = true;
    for (int i = 0; i < numOfElements; i++) {
        float diff = fabs(C[i] - C_seq[i]);
        if (diff > 0.1f) {
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
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);

    free(A_seq);
    free(B_seq);
    free(C_seq);

    free(source);
    free(platforms);
    free(devices);
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
    int platformId;
    if (argc > 2) {
        platformId = atoi(argv[1]);
        elements1D = atoi(argv[2]);
        width = elements1D; // Square matrices (MxM)
        numOfElements = elements1D * width;
        datasize = sizeof(float) * numOfElements;
    } else {
        cout << "Run: ./host <platformId> <elements>" << endl;
        return -1;
    }

    cout << "OpenCL Matrix Multiplication" << endl;
    cout << "Number of Elements: " << numOfElements << endl;

    // Discover platforms and select one
    if (discoverPlatformsAndSelect(platformId) != CL_SUCCESS) {
        return -1;
    }

    // Initialize Data Structures (Section 3.2.1)
    if (initializeDataStructures() != CL_SUCCESS) {
        return -1;
    }
    if (createDataBuffers() != CL_SUCCESS) {
        return -1;
    }

    // Initialization of input data both on the host and on the area mapped to the platform buffers
    initializeInputDataOnHostAndPlatform();

    // Transfer input data (Section 3.2.2)
    transferInputDataToPlatform();
    writeTime = getTimeInNanoseconds(writeEvent1);
    writeTime += getTimeInNanoseconds(writeEvent2);

    // Create kernel object from program source (Section 3.2.3)
    if (createKernelFromProgramSource(programSourceFileName) != CL_SUCCESS) {
        return -1;
    }
    if (setKernelArgs() != CL_SUCCESS) {
        return -1;
    }
    if (launchKernel() != CL_SUCCESS) {
        return -1;
    }
    kernelTime = getTimeInNanoseconds(kernelEvent);

    // Transfer result data (Section 3.2.4)
    transferResultDataFromPlatform();
    readTime = getTimeInNanoseconds(readEvent);

    // Report the timing from OpenCL events
    cout << "\nTransferring Input Data Time   : " << writeTime << " nanoseconds." << endl;
    cout << "Kernel Execution Time          : " << kernelTime << " nanoseconds." << endl;
    cout << "Transferring Result Data Time  : " << readTime << " nanoseconds." << endl;
    cout << "\n";

    matrixMultiplication(A_seq, B_seq, C_seq, elements1D, elements1D, width);

    // Validate the results of the kernel
    validateResultsOfKernel(C, C_seq);

    freeMemory();

    return 0;
}
