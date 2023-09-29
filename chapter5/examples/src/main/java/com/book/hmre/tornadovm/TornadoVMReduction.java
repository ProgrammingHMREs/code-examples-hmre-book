package com.book.hmre.tornadovm;

import com.book.hmre.common.Utils;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.annotations.Reduce;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;

import java.util.Random;
import java.util.stream.IntStream;

/**
 * How to run from the command line:
 * <code>
 *     tornado -cp target/examples-1.0-SNAPSHOT.jar com.book.hmrs.tornadovm.TornadoVMReduction
 * </code>
 *
 * <p>
 *     With profiling information enabled:
 *     <code>
 *         tornado --jvm="-Dcompute.reduce.device=0:0" --printKernel --threadInfo -cp target/examples-1.0-SNAPSHOT.jar com.book.hmrs.tornadovm.TornadoVMReduction
 *     </code>
 * </p>
 */
public class TornadoVMReduction {

    private static final int ITERATIONS = 15;
    private static final boolean VALIDATION = true;
    private static boolean RUN_SEQUENTIAL = true;

    /**
     * Kernel to be offloaded to the Accelerator (e.g., a GPU).
     * @param input input data set
     * @param output output array
     */
    private static void reduction(float[] input, @Reduce float[] output) {
        for (@Parallel int i = 0; i < input.length; i++) {
            output[0] += input[i];
        }
    }

    public static void main(String[] args) {

        // Data preparation
        final int SIZE = 1024;
        float[] inputArray = new float[SIZE];

        // The output is an array with a single element.
        // TornadoVM will perform a full reduction
        float[] outputArray = new float[1];

        Random r = new Random();
        IntStream.range(0, SIZE).forEach(i-> inputArray[i] = r.nextFloat());

        // Create TornadoVM TaskGraph
        TaskGraph taskGraph =  new TaskGraph("compute");
        taskGraph.transferToDevice(DataTransferMode.FIRST_EXECUTION, inputArray)
                .task("reduce", TornadoVMReduction::reduction, inputArray, outputArray)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, outputArray);

        // Create an Immutable Task Graph
        ImmutableTaskGraph itg = taskGraph.snapshot();

        // Create an Execution Plan from all immutable task graphs
        TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(itg);

        // Enable profiler metrics
        executionPlan.withProfiler(ProfilerMode.SILENT);


        long[] parallelTotalTimers = new long[ITERATIONS];
        long[] sequentialTotalTimers = new long[ITERATIONS];
        float sequentialOutputResult = 0.0f;

        // Execute the application on the default accelerator
        for (int i = 0; i < ITERATIONS; i++ ){

            TornadoExecutionResult executionResult = executionPlan.execute();
            long kernelTime = executionResult.getProfilerResult().getDeviceKernelTime();
            parallelTotalTimers[i] = kernelTime;
            System.out.print("Parallel Kernel Time (ns) = " + parallelTotalTimers[i]);

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