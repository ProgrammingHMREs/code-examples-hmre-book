package com.book.hmre.tornadovm;

import com.book.hmre.common.Utils;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.collections.types.Matrix2DFloat;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;

import java.util.Random;

/**
 * How to run from the command line:
 * <code>
 *     tornado -cp target/examples-1.0-SNAPSHOT.jar com.book.hmre.tornadovm.TornadoVMMxM
 * </code>
 *
 * <p>
 *     With profiling information enabled:
 *     <code>
 *         tornado --jvm="-Dcompute.mxm.device=0:0" --printKernel --threadInfo -cp target/examples-1.0-SNAPSHOT.jar com.book.hmre.tornadovm.TornadoVMMxM
 *     </code>
 * </p>
 */
public class TornadoVMMxM {

    private static final int ITERATIONS = 15;
    private static final boolean RUN_SEQUENTIAL = true;
    private static boolean VALIDATION = true;

    /**
     * Matrix Multiplication expressed with TornadoVM.
     *
     * @param matrixA
     * @param matrixB
     * @param matrixC
     */
    private static void matrixMultiplication(Matrix2DFloat matrixA, Matrix2DFloat matrixB, Matrix2DFloat matrixC) {
        for (@Parallel int i = 0; i < matrixA.getNumRows(); i++) {
            for (@Parallel int j = 0; j < matrixB.getNumRows(); j++) {
                float sum = 0.0f;
                for (int k = 0; k < matrixC.getNumRows(); k++) {
                    sum += matrixA.get(i, k) * matrixB.get(k, j);
                }
                matrixC.set(i, j, sum);
            }
        }
    }

    public static void main(String[] args) {

        // Data preparation
        final int SIZE = 1024;
        Matrix2DFloat matrixA = new Matrix2DFloat(SIZE, SIZE);
        Matrix2DFloat matrixB = new Matrix2DFloat(SIZE, SIZE);
        Matrix2DFloat matrixC = new Matrix2DFloat(SIZE, SIZE);
        float[] resultC = new float[SIZE * SIZE];

        // Initialize data
        Random r = new Random();
        for (int i = 0; i < matrixA.getNumRows(); i++) {
            for (int j = 0; j < matrixA.getNumColumns(); j++) {
                matrixA.set(i, j, r.nextFloat());
                matrixB.set(i, j, r.nextFloat());
            }
        }

        // Create TornadoVM TaskGraph
        TaskGraph taskGraph = new TaskGraph("compute");
        taskGraph.transferToDevice(DataTransferMode.EVERY_EXECUTION, matrixA, matrixB)
                .task("mxm", TornadoVMMxM::matrixMultiplication, matrixA, matrixB, matrixC)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, matrixC);

        // Create an Immutable Task Graph
        ImmutableTaskGraph itg = taskGraph.snapshot();

        // Create an Execution Plan from all immutable task graphs
        TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(itg);

        // Execute the application on the default accelerator
        // This is a blocking call, and it waits until the execution on the accelerator finishes.
        executionPlan.withProfiler(ProfilerMode.SILENT);

        long[] parallelTotalTimers = new long[ITERATIONS];
        long[] sequentialTotalTimers = new long[ITERATIONS];

        for (int i = 0; i < ITERATIONS; i++) {
            TornadoExecutionResult executionResult = executionPlan.execute();
            long kernelTime = executionResult.getProfilerResult().getDeviceKernelTime();
            parallelTotalTimers[i] = kernelTime;
            System.out.print("Parallel Kernel Time (ns) = " + parallelTotalTimers[i]);

            if (RUN_SEQUENTIAL) {
                long startTime = System.nanoTime();
                for (int ic = 0; ic < SIZE; ic++) {
                    for (int j = 0; j < SIZE; j++) {
                        float acc = 0.0f;
                        for (int k = 0; k < SIZE; k++) {
                            acc += matrixA.get(ic, k) * matrixB.get(k, j);
                        }
                        resultC[ic * SIZE + j] = acc;
                    }
                }
                long endTime = System.nanoTime();
                sequentialTotalTimers[i] = endTime - startTime;
                System.out.println("\tJava Sequential Time (ns) = " + sequentialTotalTimers[i]);
            }
        }

        // Statistics
        System.out.println("TornadoVM TIMERS");
        Utils.computeStatistics(parallelTotalTimers);
        System.out.println("\n\nJAVA TIMERS");
        Utils.computeStatistics(sequentialTotalTimers);

        if (VALIDATION) {
            boolean isCorrect = true;
            for (int i = 0; i < SIZE; i++) {
                for (int j = 0; j < SIZE; j++) {
                    if (Math.abs(matrixC.get(i, j) - resultC[i * SIZE + j]) > 0.1) {
                        System.out.println(matrixC.get(i, j) + " vs " + resultC[i * SIZE + j]);
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