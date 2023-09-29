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

import java.util.Arrays;
import java.util.LongSummaryStatistics;
import java.util.Random;
import java.util.stream.IntStream;

/**
 * Example to perform a reduction in JOCL.
 */
public class ReductionSample {

    private static final int ITERATONS = 15;

    private static final boolean VALIDATION = true;

    private static final boolean RUN_SEQUENTIAL = true;

    /**
     * Reduction in JOCL.
     */
    private static String openCLReductionProgram =
            "__kernel void "+
                    "reduce(__global const float *input," +
                    "       __global float *partialSums," +
                    "       __local float *localSums)" +
                    "{"+
                    "    uint idx = get_global_id(0);" +
                    "    uint localIdx = get_local_id(0);" +
                    "    uint group_size = get_local_size(0);" +
                    "    localSums[localIdx] = input[idx]; " +
                    "    for (int stride = group_size/2; stride > 0; stride /= 2) {"+
                    "        barrier(CLK_LOCAL_MEM_FENCE);" +
                    "        localSums[localIdx] += localSums[localIdx + stride];"+
                    "    }"+
                    "    if (localIdx == 0) {"+
                    "         partialSums[get_group_id(0)] = localSums[0]; " +
                    "    };"+
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
        long[] size = new long[1];
        CL.clGetPlatformInfo(platform, parameterName, 0, null, size);

        // Create a buffer of the appropriate size and fill it with the info
        byte[] buffer = new byte[(int)size[0]];
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
        final int NUMBER_OF_OPENCL_GROUPS = 4;

        // Allocate host data (data that will reside on the Java heap)
        float[] inputArray = new float[SIZE];
        float[] outputArray = new float[NUMBER_OF_OPENCL_GROUPS];
        Pointer hostPtrInput = Pointer.to(inputArray);
        Pointer hostPtrOutput = Pointer.to(outputArray);

        // Data Initialization
        Random r = new Random();
        IntStream.range(0, SIZE).forEach(i -> inputArray[i] = r.nextFloat());

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
        final long datasizeInput = Sizeof.cl_float * SIZE;
        final long dataSizeOutput = Sizeof.cl_float * NUMBER_OF_OPENCL_GROUPS;
        cl_mem deviceBufferInput = CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY, datasizeInput, null, null);
        cl_mem deviceBufferOutput = CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY, dataSizeOutput, null, null);

        // Create the program from the source code
        cl_program program = CL.clCreateProgramWithSource(context, 1, new String[]{openCLReductionProgram}, null, null);

        // Build the program
        CL.clBuildProgram(program, 0, null, null, null, null);

        // Create the kernel
        cl_kernel kernel = CL.clCreateKernel(program, "reduce", null);

        long[] parallelTotalTimers = new long[ITERATONS];
        long[] sequentialTotalTimers = new long[ITERATONS];
        float sequentialOutputResult = 0.0f;

        for (int i = 0; i < ITERATONS; i++) {

            // Send data
            CL.clEnqueueWriteBuffer(commandQueue, deviceBufferInput, CL.CL_TRUE, 0, datasizeInput, hostPtrInput, 0, null, null);

            // Set the arguments for the kernel
            int argIndex = 0;
            CL.clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(deviceBufferInput));
            CL.clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(deviceBufferOutput));
            CL.clSetKernelArg(kernel, argIndex++, Sizeof.cl_float * 256, null);

            long[] global_work_size = new long[]{SIZE, 1, 1};
            long[] lobal_work_size = new long[]{256, 1, 1};

            cl_event kernelEvent = new cl_event();

            // Execute the kernel
            CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, global_work_size, lobal_work_size, 0, null, kernelEvent);

            // Wait for the kernel to finish before collecting profiling info
            CL.clWaitForEvents(1, new cl_event[]{kernelEvent});

            // Read the output data
            CL.clEnqueueReadBuffer(commandQueue, deviceBufferOutput, CL.CL_TRUE, 0, dataSizeOutput, hostPtrOutput, 0, null, null);

            long[] start = new long[1];
            CL.clGetEventProfilingInfo(kernelEvent, CL.CL_PROFILING_COMMAND_START, Sizeof.cl_long, Pointer.to(start), null);
            long[] end = new long[1];
            CL.clGetEventProfilingInfo(kernelEvent, CL.CL_PROFILING_COMMAND_END, Sizeof.cl_long, Pointer.to(end), null);
            parallelTotalTimers[i] = end[0] - start[0];
            System.out.print("Parallel Kernel Time (ns) = " + parallelTotalTimers[i]);

            // Second part of the reduction:
            for (int k = 1; k < NUMBER_OF_OPENCL_GROUPS; k++) {
                outputArray[0] += outputArray[k];
            }

            if (RUN_SEQUENTIAL) {
                long startTime = System.nanoTime();
                sequentialOutputResult = 0.0f;
                for (int k = 0; k < SIZE; k++) {
                    sequentialOutputResult += inputArray[k];
                }
                long endTime = System.nanoTime();
                sequentialTotalTimers[i] = endTime - startTime;
                System.out.println("\tJava Sequential Time (ns) = " + sequentialTotalTimers[i]);
            }


        }

        // Release kernel, program, and memory objects
        CL.clReleaseMemObject(deviceBufferInput);
        CL.clReleaseMemObject(deviceBufferOutput);
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
            if (Math.abs(outputArray[0] - sequentialOutputResult) > 0.1) {
                System.out.println(outputArray[0] + " != " + sequentialOutputResult);
                isCorrect = false;
            }

            if (isCorrect) {
                System.out.println("\nResult is correct");
            } else {
                System.out.println("\nResult is wrong");
            }
        }
    }
}
