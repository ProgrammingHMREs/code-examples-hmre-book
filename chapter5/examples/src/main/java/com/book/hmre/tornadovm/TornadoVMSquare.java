package com.book.hmre.tornadovm;

import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

import java.util.Random;
import java.util.stream.IntStream;

/**
 * How to run from the command line:
 *
 * <p>
 * <code>
 *     tornado -cp target/examples-1.0-SNAPSHOT.jar com.book.hmrs.tornadovm.TornadoVMSquare
 * </code>
 * </p>
 *
 * <p>
 * Enable profiling information:
 * <code>
 *     tornado --jvm="-Dcompute.square.device=0:0" --printKernel --threadInfo -cp target/examples-1.0-SNAPSHOT.jar com.book.hmrs.tornadovm.TornadoVMSquare
 * </code>
 * </p>
 *
 */
public class TornadoVMSquare {

    private static void computeSquare(float[] input, float[] output) {
        for (@Parallel int i = 0; i < input.length; i++) {
            output[i] = input[i] * input[i];
        }
    }

    private static void computeSquare(float[] input, float[] output, KernelContext context) {
        int idx = context.globalIdx;
        output[idx] = input[idx] * input[idx];
    }

    public static void main(String[] args) {
        System.out.println("Hello TornadoVM\n");

        // Data preparation
        final int SIZE = 100000;
        float[] inputArray = new float[SIZE];
        float[] outputArray = new float[SIZE];
        Random r = new Random();
        IntStream.range(0, SIZE).forEach(i-> inputArray[i] = r.nextFloat());

        // Create TornadoVM TaskGraph
        // Definition of data and code to be executed
        TaskGraph taskGraph =  new TaskGraph("compute");
        taskGraph.transferToDevice(DataTransferMode.FIRST_EXECUTION, inputArray)
                .task("square", TornadoVMSquare::computeSquare, inputArray, outputArray)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, outputArray);

        // Create an Immutable Task Graph
        ImmutableTaskGraph itg = taskGraph.snapshot();

        // Create an Execution Plan from all immutable task graphs
        TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(itg);

        // Execute the execution plan on the default device
        executionPlan.execute();

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
