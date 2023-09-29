__kernel void matrixMultiplication(__global float* A, __global float* B, __global float* C, int sizeX, int sizeY, int width) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    float sum = 0.0f;
    for (int k = 0; k < width; k++) {
        sum += A[k * sizeX + i] * B[j * width + k];
    }
    C[j * sizeX + i] = sum;
}