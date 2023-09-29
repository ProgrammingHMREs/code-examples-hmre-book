package com.book.hmre.jocl;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_device_id;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_platform_id;
import org.jocl.cl_program;
import org.jocl.cl_queue_properties;

import java.util.Random;
import java.util.stream.IntStream;

/**
 * Simple Test Application for JOCL
 */
public class TestSquare {

    /**
     * OpenCL C code for the Square Computation
     */
    private static String openclSquareProgram =
            "__kernel void "+
                    "square(__global const float *a,"+
                    "       __global float *b," +
                    "       const int size)" +
                    "{"+
                    "    int gid = get_global_id(0);"+
                    "    if (gid < size) {"+
                    "        b[gid] = a[gid] * a[gid];"+
                    "    }"+
                    "}";


    public static cl_platform_id getOpenCLPlatform() {
        // Obtain the OpenCL Platform
        // First we need the number of platforms
        int[] numPlatformsArray = new int[1];
        CL.clGetPlatformIDs(0, null, numPlatformsArray);
        final int numPlatforms = numPlatformsArray[0];

        // Second, we obtain the platform ids
        cl_platform_id[] platforms = new cl_platform_id[numPlatforms];
        CL.clGetPlatformIDs(platforms.length, platforms, null);

        // Get access to the desired platform
        return platforms[0];
    }

    public static cl_device_id getOpenCLDevice(cl_platform_id platform, int deviceIndex) {

        int[] numDevicesArray = new int[1];
        CL.clGetDeviceIDs(platform, CL.CL_DEVICE_TYPE_ALL, 0, null, numDevicesArray);
        final int numDevices = numDevicesArray[0];

        cl_device_id[] devices = new cl_device_id[numDevices];
        CL.clGetDeviceIDs(platform, CL.CL_DEVICE_TYPE_ALL, numDevices, devices, null);
        return devices[deviceIndex];
    }

    public static void main( String[] args ) {

        final int SIZE = 100000;

        // Allocate host data (data that will reside on the Java heap)
        float[] inputArray = new float[SIZE];
        float[] outputArray = new float[SIZE];
        Pointer srcA = Pointer.to(inputArray);
        Pointer srcB = Pointer.to(outputArray);

        Random r = new Random(31);
        // Initialize the data
        IntStream.range(0, SIZE).forEach( i-> inputArray[i] = r.nextFloat());

        // Enable exceptions, and therefore, omit error checks in this sample
        CL.setExceptionsEnabled(true);

        cl_platform_id platform = getOpenCLPlatform();
        cl_device_id device = getOpenCLDevice(platform, 0);

        // Create a context for the selected device
        cl_context context = CL.clCreateContext(null, 1, new cl_device_id[]{device},null, null, null);

        // create the command queue
        cl_queue_properties queueProperties = new cl_queue_properties();
        cl_command_queue commandQueue = CL.clCreateCommandQueue(context, device, 0, null);

        // Create OpenCL Buffers
        cl_mem bufferA = CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY | CL.CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * SIZE, srcA, null);
        cl_mem bufferB = CL.clCreateBuffer(context, CL.CL_MEM_WRITE_ONLY | CL.CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * SIZE, srcB, null);


        // Alternative to create and copy a buffer: The next two calls are used for illustration purposes.
        cl_mem deviceBuffer = CL.clCreateBuffer(context,
                CL.CL_MEM_WRITE_ONLY,
                Sizeof.cl_float * SIZE,
                null,
                null);
        int status = CL.clEnqueueWriteBuffer(commandQueue,
                deviceBuffer,
                CL.CL_TRUE,  // blocking operation
                0,
                Sizeof.cl_float * SIZE,
                Pointer.to(inputArray),
                0,
                null,
                null);
        if (status != CL.CL_SUCCESS) {
            System.out.println("Error: " + status);
        }

        // Create the program from the source code
        cl_program program = CL.clCreateProgramWithSource(context, 1, new String[]{ openclSquareProgram }, null, null);

        // Build the program
        CL.clBuildProgram(program, 0, null, null, null, null);

        // Create the kernel
        cl_kernel kernel = CL.clCreateKernel(program, "square", null);

        // Set the arguments for the kernel
        int argIndex = 0;
        CL.clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(bufferA));
        CL.clSetKernelArg(kernel, argIndex++, Sizeof.cl_mem, Pointer.to(bufferB));
        CL.clSetKernelArg(kernel, argIndex++, Sizeof.cl_uint, Pointer.to(new int[]{SIZE}));


        // Set the work-item dimensions
        long global_work_size[] = new long[]{SIZE};

        // Execute the kernel
        CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, global_work_size, null, 0, null, null);

        // Read the output data
        CL.clEnqueueReadBuffer(commandQueue, bufferB, CL.CL_TRUE, 0,SIZE * Sizeof.cl_float, srcB, 0, null, null);

        // Release kernel, program, and memory objects
        CL.clReleaseMemObject(bufferA);
        CL.clReleaseMemObject(bufferB);
        CL.clReleaseKernel(kernel);
        CL.clReleaseProgram(program);
        CL.clReleaseCommandQueue(commandQueue);
        CL.clReleaseContext(context);

        // Verify the result
        boolean passed = true;
        final float epsilon = 1e-7f;
        for (int i=0; i< SIZE; i++) {
            float gpuCompute = outputArray[i];
            float cpuCompute = inputArray[i] * inputArray[i];
            boolean epsilonEqual = Math.abs(cpuCompute - gpuCompute) <= epsilon;
            if (!epsilonEqual) {
                passed = false;
                break;
            }
        }
        System.out.println("Test "+(passed?"PASSED":"FAILED"));
    }
}
