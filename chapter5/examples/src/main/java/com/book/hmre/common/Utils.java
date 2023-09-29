package com.book.hmre.common;

import java.util.Arrays;
import java.util.LongSummaryStatistics;

public class Utils {

    public static void computeStatistics(long[] totalTimes) {
        LongSummaryStatistics longSummaryStatistics = Arrays.stream(Arrays.stream(totalTimes).toArray()).summaryStatistics();
        double average = longSummaryStatistics.getAverage();
        double count = longSummaryStatistics.getCount();

        double[] variance = new double[totalTimes.length];
        for (int i = 0; i < variance.length; i++) {
            variance[i] = Math.pow((totalTimes[i] - average) , 2);
        }
        double varianceScalar = Arrays.stream(variance).sum() / count;
        double std = Math.sqrt(varianceScalar);

        System.out.println("Min     : " + longSummaryStatistics.getMin());
        System.out.println("Max     : " + longSummaryStatistics.getMax());
        System.out.println("Average : " + longSummaryStatistics.getAverage());
        System.out.println("Variance: " + varianceScalar);
        System.out.println("STD     : " + std);
    }
}
