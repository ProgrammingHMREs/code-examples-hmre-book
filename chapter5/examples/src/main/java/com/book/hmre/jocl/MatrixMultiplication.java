package com.book.hmre.jocl;

import com.book.hmre.common.Utils;
import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_device_id;
import org.jocl.cl_event;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_platform_id;
import org.jocl.cl_program;

import java.util.Random;

/**
 * Classic algorithm for the Matrix Multiplication implemented in OpenCL.
 */
public class MatrixMultiplication {

    private static final int ITERATONS = 15;

    private static final boolean VALIDATION = true;

    public static final boolean RUN_SEQUENTIAL = true;
    /**
     * Matrix Multiplication OpenCL C program.
     */
    private static String openclMatrixMultiplicationProgram =
            "__kernel void "+
                    "mxm(__global const float *a,"+
                    "    __global const float *b," +
                    "    __global float *c," +
                    "     const int size)" +
                    "{"+
                    "    int idx = get_global_id(0);"+
                    "    int idy = get_global_id(1);"+
                    "    float sum = 0.0f;"+
                    "    for (int k = 0; k < size; k++) {"+
                    "        sum += a[idx * size + k] * b[k * size + idy];"+
                    "    }"+
                    "    c[idx * size + idy] = sum;"+
                    "}";


    public static cl_platform_id getOpenCLPlatform(int platformIndex) {
        // Obtain the OpenCL Platform
        // First we need the number of platforms
        int[] numPlatformsArray = new int[1];
        CL.clGetPlatformIDs(0, null, numPlatformsArray);
        final int numPlatforms = numPlatformsArray[0];

        // Second, we obtain the platform ids
        cl_platform_id[] platforms = new cl_platform_id[numPlatforms];
        CL.clGetPlatformIDs(platforms.length, platforms, null);

        // Get access to the desired platform
        return platforms[platformIndex];
    }

    /**
     * Utility borrowed from JOCL code samples. {@url https://github.com/gpu/JOCLSamples/blob/master/src/main/java/org/jocl/samples/JOCLDeviceQuery.java#L277}
     *
     * Returns the value of the platform info parameter with the given name
     *
     * @param platform The platform
     * @param parameterName The parameter name
     * @return The value
     */
    private static String getString(cl_platform_id platform, int parameterName)
    {
        // Obtain the length of the string that will be queried
        long size[] = new long[1];
        CL.clGetPlatformInfo(platform, parameterName, 0, null, size);

        // Create a buffer of the appropriate size and fill it with the info
        byte buffer[] = new byte[(int)size[0]];
        CL.clGetPlatformInfo(platform, parameterName, buffer.length, Pointer.to(buffer), null);

        // Create a string from the buffer (excluding the trailing \0 byte)
        return new String(buffer, 0, buffer.length-1);
    }

    /**
     * Utility borrowed from JOCL code samples. {@url https://github.com/gpu/JOCLSamples/blob/master/src/main/java/org/jocl/samples/JOCLDeviceQuery.java#L249C1-L268C6}
     *
     * Returns the value of the device info parameter with the given name
     *
     * @param device The device
     * @param paramName The parameter name
     * @return The value
     */
    private static String getString(cl_device_id device, int paramName)
    {
        // Obtain the length of the string that will be queried
        long size[] = new long[1];
        CL.clGetDeviceInfo(device, paramName, 0, null, size);

        // Create a buffer of the appropriate size and fill it with the info
        byte buffer[] = new byte[(int)size[0]];
        CL.clGetDeviceInfo(device, paramName, buffer.length, Pointer.to(buffer), null);

        // Create a string from the buffer (excluding the trailing \0 byte)
        return new String(buffer, 0, buffer.length-1);
    }

    public static cl_device_id getOpenCLDevice(cl_platform_id platform, int deviceIndex) {

        int[] numDevicesArray = new int[1];
        CL.clGetDeviceIDs(platform, CL.CL_DEVICE_TYPE_GPU, 0, null, numDevicesArray);
        final int numDevices = numDevicesArray[0];

        cl_device_id[] devices = new cl_device_id[numDevices];
        CL.clGetDeviceIDs(platform, CL.CL_DEVICE_TYPE_GPU, numDevices, devices, null);
        return devices[deviceIndex];
    }

    public static void main( String[] args ) {

        final int SIZE = 1024;

        // Allocate host data (data that will reside on the Java heap)
        float[] matrixA = new float[SIZE * SIZE];
        float[] matrixB = new float[SIZE * SIZE];
        float[] matrixC = new float[SIZE * SIZE];
        Pointer hostPtrMatrixA = Pointer.to(matrixA);
        Pointer hostPtrMatrixB = Pointer.to(matrixB);
        Pointer hostPtrMatrixC = Pointer.to(matrixC);

        // Data Initialization
        Random r = new Random();
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                matrixA[i * SIZE + j] = r.nextFloat();
                matrixB[i * SIZE + j] = r.nextFloat();
            }
        }

        // Enable exceptions, and therefore, omit error checks in this sample
        CL.setExceptionsEnabled(true);

        System.out.println("----------------------------------");
        cl_platform_id platform = getOpenCLPlatform(1);
        String platformName = getString(platform, CL.CL_PLATFORM_NAME);
        System.out.println("Using OpenCL platform: " + platformName);
        cl_device_id device = getOpenCLDevice(platform, 0);
        String deviceName = getString(device, CL.CL_DEVICE_NAME);
        System.out.println("Device Name: " + deviceName);
        System.out.println("----------------------------------");

        // Create a context for the selected device
        cl_context context = CL.clCreateContext(null, 1, new cl_device_id[]{device},null, null, null);

        // create the command queue
        cl_command_queue commandQueue = CL.clCreateCommandQueue(context, device, CL.CL_QUEUE_PROFILING_ENABLE, null);

        // Create OpenCL Buffers
        long datasize = Sizeof.cl_float * SIZE * SIZE;
        cl_mem deviceBufferA = CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY, datasize, null, null);
        cl_mem deviceBufferB = CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY, datasize, null, null);
        cl_mem deviceBufferC = CL.clCreateBuffer(context, CL.CL_MEM_WRITE_ONLY, datasize, null, null);

        // Create the program from the source code
        cl_program program = CL.clCreateProgramWithSource(context, 1, new String[]{openclMatrixMultiplicationProgram}, null, null);

        // Build the program
        CL.clBuildProgram(program, 0, null, null, null, null);

        // Create the kernel
        cl_kernel kernel = CL.clCreateKernel(program, "mxm", null);

        long[] parallelTotalTimers = new long[ITERATONS];
        long[] sequentialTotalTimers = new long[ITERATONS];
        float[] resultC = new float[SIZE * SIZE];

        for (int i = 0; i < ITERATONS; i++) {

            // Send data
            CL.clEnqueueWriteBuffer(commandQueue, deviceBufferA, CL.CL_TRUE, 0, datasize, hostPtrMatrixA, 0, null, null);
            CL.clEnqueueWriteBuffer(commandQueue, deviceBufferB, CL.CL_TRUE, 0, datasize, hostPtrMatrixB, 0, null, null);

            // Set the arguments for the kernel
            int argIndex = 0;
            CL.clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(deviceBufferA));
            CL.clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(deviceBufferB));
            CL.clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(deviceBufferC));
            CL.clSetKernelArg(kernel, argIndex++, Sizeof.cl_uint, Pointer.to(new int[]{SIZE}));

            // Set the work-item dimensions
            long[] global_work_size = new long[]{SIZE, SIZE};

            cl_event kernelEvent = new cl_event();

            // Execute the kernel
            CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, global_work_size, null, 0, null, kernelEvent);

            // Wait for the kernel to finish before collecting profiling info
            CL.clWaitForEvents(1, new cl_event[]{kernelEvent});

            long[] start = new long[1];
            CL.clGetEventProfilingInfo(kernelEvent, CL.CL_PROFILING_COMMAND_START, Sizeof.cl_long, Pointer.to(start), null);
            long[] end = new long[1];
            CL.clGetEventProfilingInfo(kernelEvent, CL.CL_PROFILING_COMMAND_END, Sizeof.cl_long, Pointer.to(end), null);
            parallelTotalTimers[i] = end[0] - start[0];
            System.out.print("Parallel Kernel Time (ns) = " + parallelTotalTimers[i]);

            // Read the output data
            CL.clEnqueueReadBuffer(commandQueue, deviceBufferC, CL.CL_TRUE, 0, datasize, hostPtrMatrixC, 0, null, null);

            if (RUN_SEQUENTIAL) {
                long startTime = System.nanoTime();
                for (int ic = 0; ic < SIZE; ic++) {
                    for (int j = 0; j < SIZE; j++) {
                        float acc = 0.0f;
                        for (int k = 0; k < SIZE; k++) {
                            acc += matrixA[ic * SIZE + k] * matrixB[k * SIZE + j];
                        }
                        resultC[ic * SIZE + j] = acc;
                    }
                }
                long endTime = System.nanoTime();
                sequentialTotalTimers[i] = endTime - startTime;
                System.out.println("\tJava Sequential Time (ns) = " + sequentialTotalTimers[i]);
            }
        }

        // Release kernel, program, and memory objects
        CL.clReleaseMemObject(deviceBufferA);
        CL.clReleaseMemObject(deviceBufferB);
        CL.clReleaseMemObject(deviceBufferC);
        CL.clReleaseKernel(kernel);
        CL.clReleaseProgram(program);
        CL.clReleaseCommandQueue(commandQueue);
        CL.clReleaseContext(context);

        System.out.println("OpenCL program finished");

        // Statistics
        System.out.println("JOCL TIMERS");
        Utils.computeStatistics(parallelTotalTimers);
        System.out.println("\n\nJAVA TIMERS");
        Utils.computeStatistics(sequentialTotalTimers);

        if (VALIDATION) {
            boolean isCorrect = true;
            for (int i = 0; i < SIZE; i++) {
                for (int j = 0; j < SIZE; j++) {
                    if (Math.abs(matrixC[i * SIZE + j] - resultC[i * SIZE + j]) > 0.1) {
                        System.out.println(matrixC[i * SIZE + j] + " vs " + resultC[i * SIZE + j]);
                        isCorrect = false;
                        break;
                    }
                }
                if (!isCorrect) {
                    break;
                }
            }

            if (isCorrect) {
                System.out.println("\nResult is correct");
            } else {
                System.out.println("\nResult is wrong");
            }
        }
    }
}
