#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

_global_ void suma_gpu(int* num1, int* num2, int* resultado)
{
    *resultado = *num1 + *num2;
}

void suma_cpu(int* num1, int* num2, int* resultado)
{
    *resultado = *num1 + *num2;
}

_global_ void resta_gpu(int* num1, int* num2, int* resultado)
{
    *resultado = *num2 - *num1;
}

void resta_cpu(int* num1, int* num2, int* resultado)
{
    *resultado = *num2 - *num1;
}

_global_ void multiplicacion_gpu(int* num1, int* num2, int* resultado)
{
    *resultado = *num1 * *num2;
}

void multiplicacion_cpu(int* num1, int* num2, int* resultado)
{
    *resultado = *num1 * *num2;
}

_global_ void division_gpu(int* num1, int* num2, int* resultado)
{
    if (*num1 != 0)
        *resultado = *num2 / *num1;
    else
        *resultado = 0;
}

void division_cpu(int* num1, int* num2, int* resultado)
{
    if (*num1 != 0)
        *resultado = *num2 / *num1;
    else
        *resultado = 0;
}

int main()
{
    int num1 = 2;
    int num2 = 5;
    int res;

    printf("CPU:\n");
    suma_cpu(&num1, &num2, &res);
    printf("Suma: %d\n", res);

    resta_cpu(&num1, &num2, &res);
    printf("Resta: %d\n", res);

    multiplicacion_cpu(&num1, &num2, &res);
    printf("Multiplicacion: %d\n", res);

    division_cpu(&num1, &num2, &res);
    printf("Division: %d\n", res);

    int* num1_gpu, * num2_gpu, * resultado_gpu;
    int tamano = sizeof(int);

    cudaMalloc((void**)&num1_gpu, tamano);
    cudaMalloc((void**)&num2_gpu, tamano);
    cudaMalloc((void**)&resultado_gpu, tamano);

    cudaMemcpy(num1_gpu, &num1, tamano, cudaMemcpyHostToDevice);
    cudaMemcpy(num2_gpu, &num2, tamano, cudaMemcpyHostToDevice);

    printf("\nGPU:\n");
    suma_gpu << <1, 1 >> > (num1_gpu, num2_gpu, resultado_gpu);
    cudaMemcpy(&res, resultado_gpu, tamano, cudaMemcpyDeviceToHost);
    printf("Suma: %d\n", res);

    resta_gpu << <1, 1 >> > (num1_gpu, num2_gpu, resultado_gpu);
    cudaMemcpy(&res, resultado_gpu, tamano, cudaMemcpyDeviceToHost);
    printf("Resta: %d\n", res);

    multiplicacion_gpu << <1, 1 >> > (num1_gpu, num2_gpu, resultado_gpu);
    cudaMemcpy(&res, resultado_gpu, tamano, cudaMemcpyDeviceToHost);
    printf("Multiplicacion: %d\n", res);

    division_gpu << <1, 1 >> > (num1_gpu, num2_gpu, resultado_gpu);
    cudaMemcpy(&res, resultado_gpu, tamano, cudaMemcpyDeviceToHost);
    printf("Division: %d\n", res);

    cudaFree(num1_gpu);
    cudaFree(num2_gpu);
    cudaFree(resultado_gpu);

    return 0;
}