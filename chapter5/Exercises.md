## Chapter 5 | Exercises

1. In relation to the wrappers programming approach to implement applications using managed runtime programming environments to access hardware accelerators, why non-blocking calls are not recommended?

	a. If there are any issues, is there any way to work around those issues? 
	
	b. How do off-heap memory data types solve the issue?
	
	c. List examples of off-heap memory Java APIs.
	
2. Change the OpenCL C program for the Matrix Multiplication implemented in JOCL for launching and accessing a 1D kernels.
	a. What are the changes required?
	b. Is there any performance difference in the kernel execution time? If so, why is this performance difference?
	
3. Adapt the code example in JOCL explained in Section 5.2.2 to perform buffer allocation and data transfers in separated JOCL calls. You can expand the source code for this example available here: https://github.com/ProgrammingHMREs/code-examples-hmre-book/tree/main/chapter5/examples.

4. Expand the Matrix-Multiplication implemented in Java and JOCL that we ex- plained in Section 5.2.3 for non-square matrices. You can start with the source code on GitHub for the squared version of the Matrix Multiplication and add the new changes in the class MatrixMultiplication.java.

5. Measure the time that takes to perform the data transfers (copies from the host to the device and from the device to the host) for the Matrix-Multiplication JOCL example that we explained in Section 5.2.3.
	a. Measure the kernel time versus the Java execution time. Is there any difference? Which one is faster? How is the complexity and the amount of operations compared to the sum reduction explained in Section 5.5.3.
	
6. Optimize the JOCL code for the matrix multiplication explained in Section 5.2.3 to perform loop interchange [87] and compare the performance for different matrix sizes with the equivalent application in TornadoVM explained in Section 5.4.4. 

7. Adapt the Java reduction code shown below that estimates the Pi number to run with TornadoVM.

```java
	public static void computePi(float[] input, float[] result) {
    	result[0] = 0.0f;
    	for (int i = 1; i < input.length; i++) {
        	float value = input[i] + (TornadoMath.pow(-1, i + 1) / (2 * i - 1));
        	result[0] += value;
    	}
	}
```	

	a. Enable the profiler with the option −−enableProfiler console from the command line and analyze the kernel execution time and the data transfers time.
	
	b. How many kernels were generated for the TornadoVM implementation of the code snippet in Listing 63? What is the purpose of each of the generated kernels?
	
8. What are the advantages and disadvantages of each programming approach that we saw in this chapter, namely, wrapper libraries, usage of pre-built kernels and fully automatic solutions? Under which software, functional and performance requirements would you use each of them?

9. Make a table in which you describe the challenges for this approach and how common framework overcome these challenges.