#include <CL/cl.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include "timers.h"
#define EDEV 10

int print_matrix = 0;
int validation = 0;

const char* kernel_src = 
				"__kernel void mat_mul(__global float *A, " \
				"__global float *B, " \
				"__global float *C, " \
				"int NDIM)" \
				"{" \
				"	int id1 = get_global_id(0);" \
				"	int id2 = get_global_id(1);" \
				"	int i;" \
				"	float result = 0;" \
				"	for (i = 0; i < NDIM; i++) {" \
				"		result += A[id1 * NDIM + i] * B[i * NDIM + id2];" \
				"	}" \
				"	C[id1 * NDIM + id2] = result;" \
				"}";

void mat_mul( float *c, float *a, float *b, int NDIM, int NGPU, int NCPU) {
	cl_int error;
	int i;
	int device_num = NGPU + NCPU; 
	int gpu_ndev;
	int cpu_ndev;

	cl_platform_id platform;
	error = clGetPlatformIDs(1, &platform, NULL);
	if (error != CL_SUCCESS) {
		printf("Get platform id fail\n");
		exit(1);
	}

	cl_device_id device[device_num];	
	error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, (unsigned int *) &gpu_ndev);
	if (error != CL_SUCCESS) {
		printf("Get gpu number fail\n");
		exit(1);
	}
	error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, (unsigned int *) &cpu_ndev);
	if (error != CL_SUCCESS) {
		printf("Get cpu number fail\n");
		exit(1);
	}

	if (NGPU) {
		error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, NGPU, device, NULL);
		if (error != CL_SUCCESS) {
			printf("Get gpu id fail\n");
			exit(1);
		}
	}
	if (NCPU) {
		error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, NCPU, &device[device_num - 1], NULL);
		if (error != CL_SUCCESS) {
			printf("Get cpu id fail\n");
			exit(1);
		}
	}
	
	cl_context context;
	context = clCreateContext(0, device_num, device, NULL, NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Context creation failed\n");
		switch (error) {
			case CL_INVALID_PLATFORM:
				printf("Invalid platform\n");
				break;
			case CL_INVALID_VALUE:
				printf("Invalid value\n");
				break;
			case CL_INVALID_DEVICE:
				printf("Invalid device\n");
				break;
			case CL_DEVICE_NOT_AVAILABLE:
				printf("Device not available\n");
				break;
			case CL_OUT_OF_HOST_MEMORY:
				printf("Out of host mem\n");
				break;
			default:
				printf("No match!\n");
		}
		exit(1);
	}

	cl_command_queue command_queue[device_num];
	for (i = 0; i < device_num; i++) {
		command_queue[i] = clCreateCommandQueue(context, device[i], 0, &error);
		if (error != CL_SUCCESS) {
			printf("%d'th command queue creation failed\n", i);
		}
	}

	cl_mem bufferA[device_num];
	cl_mem bufferB[device_num];
	cl_mem bufferC[device_num];

	size_t hA = NDIM;
	size_t gpu_hA;
	size_t cpu_hA = 0;
	if (NCPU) {
		if (NGPU == 0) {
			cpu_hA = hA;
		} else if (NGPU == 1) {
			cpu_hA = hA / 16;
		} else if (NGPU == 2) {
			cpu_hA = hA / 32;
		} else if (NGPU == 4) {
			cpu_hA = hA / 64;
		} else {
			printf("Invalid NGPU\n");
			exit(1);
		}
	}
	gpu_hA = (hA - cpu_hA)/NGPU;
	
	size_t sizeB = NDIM * NDIM * sizeof(float);
	size_t gpu_sizeA = gpu_hA * NDIM * sizeof(float);
	size_t gpu_sizeC = gpu_sizeA;
	size_t cpu_sizeA = cpu_hA * NDIM * sizeof(float);
	size_t cpu_sizeC = cpu_sizeA;

	for (i = 0; i < device_num; i++) {
		if (i < NGPU) {
			bufferA[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, gpu_sizeA, NULL, NULL);
			bufferB[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeB, NULL, NULL);
			bufferC[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, gpu_sizeC, NULL, NULL);
		} else {
			bufferA[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, cpu_sizeA, NULL, NULL);
			bufferB[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeB, NULL, NULL);
			bufferC[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, cpu_sizeC, NULL, NULL);
		}
	}

	int NDIM_p = NDIM;

	cl_program program;
	size_t kernel_src_len = strlen(kernel_src);
	program = clCreateProgramWithSource(context, 1, (const char**) &kernel_src, &kernel_src_len, NULL);

	error = clBuildProgram(program, device_num, device, NULL, NULL, NULL);
	if (error != CL_SUCCESS) {
		printf("Program build fail\n");
		size_t len;
		char buffer[2048];
		clGetProgramBuildInfo(program, device[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);

		printf("%s\n", buffer);
		exit(1);
	}

	cl_kernel kernel[device_num];
	for (i = 0; i < device_num; i++) {
		kernel[i] = clCreateKernel(program, "mat_mul", &error);
		if (error != CL_SUCCESS) {
			printf("%d'th kerenl creation failed\n", i);
		}
	}

	for (i = 0; i < device_num; i++) {
		clSetKernelArg(kernel[i], 0, sizeof(cl_mem), (void *) &bufferA[i]);
		clSetKernelArg(kernel[i], 1, sizeof(cl_mem), (void *) &bufferB[i]);
		clSetKernelArg(kernel[i], 2, sizeof(cl_mem), (void *) &bufferC[i]);
		clSetKernelArg(kernel[i], 3, sizeof(int), (void *)&NDIM_p);
	}

	for (i = 0; i < device_num; i++) {
		if (i < NGPU) {
			error = clEnqueueWriteBuffer(command_queue[i], bufferA[i], CL_FALSE, 0, gpu_sizeA, (void *) ((size_t) a + gpu_sizeA * i), 0, NULL, NULL);
		} else {
			error = clEnqueueWriteBuffer(command_queue[i], bufferA[i], CL_FALSE, 0, cpu_sizeA, (void *) ((size_t) a + gpu_sizeA * i), 0, NULL, NULL);
		}
		if (error != CL_SUCCESS) {
			printf("%d'th A buffer enqueue failed\n", i);
		}
		error = clEnqueueWriteBuffer(command_queue[i], bufferB[i], CL_FALSE, 0, sizeB, b, 0, NULL, NULL);
		if (error != CL_SUCCESS) {
			printf("%d'th B buffer enqueue failed\n", i);
		}
	}

	size_t gpu_global[2] = {gpu_hA, NDIM};
	size_t cpu_global[2] = {cpu_hA, NDIM};
	size_t local[2] = {16, 16};
	for (i = 0; i < device_num; i++) {
		if (i < NGPU) {
			error = clEnqueueNDRangeKernel(command_queue[i], kernel[i], 2, NULL, gpu_global, local, 0, NULL, NULL);
		} else {
			error = clEnqueueNDRangeKernel(command_queue[i], kernel[i], 2, NULL, cpu_global, local, 0, NULL, NULL);
		}
		if (error != CL_SUCCESS) {
			printf("%d'th NDRange kernel enqueue failed\n", i);
		}
	}

	for (i = 0; i < device_num; i++) {
		if (i < NGPU) {
			error = clEnqueueReadBuffer(command_queue[i], bufferC[i], CL_TRUE, 0, gpu_sizeC, (void *) ((size_t) c + gpu_sizeC * i), 0, NULL, NULL);
		} else {
			error = clEnqueueReadBuffer(command_queue[i], bufferC[i], CL_TRUE, 0, cpu_sizeC, (void *) ((size_t) c + gpu_sizeC * i), 0, NULL, NULL);
		}
		if (error != CL_SUCCESS) {
			printf("%d'th C buffer enqueue failed\n", i);
		}
	}
}

/************************** DO NOT TOUCH BELOW HERE ******************************/

void check_mat_mul( float * c, float * a, float * b, int NDIM )
{
	int i, j, k, x, y;
	float sum, result;
	int validated = 1;

	printf("Validating the result..\n");
	
  srand(time(NULL));
	// C = AB
	for( x = 0; x < 10; x++ )
	{
		for( y = 0; y < 10; y++ )
		{
      i = rand() % NDIM;
      j = rand() % NDIM;
			sum = 0;

			for( k = 0; k < NDIM; k++ )
			{
				sum += a[i * NDIM + k] * b[k * NDIM + j];
			}

      result = c[i * NDIM + j];

			if( result - sum > 0.001 || result - sum < -0.001 )
			{
				printf("c[%d][%d] is differ(value=%lf correct_value=%lf)!!\n", i, j, c[i * NDIM + j], sum );
				validated = 0;
			}
		}
	}

	printf("Validation : ");
	if( validated )
		printf("SUCCESSFUL.\n");
	else
		printf("FAILED.\n");
}

void print_mat( float * mat, int NDIM )
{
	int i, j;

	for( i = 0; i < NDIM; i++ )
	{
		for( j = 0; j < NDIM; j++ )
		{
			printf("%8.2lf ", mat[i * NDIM + j]);
		}
		printf("\n");
	}
}

void print_help(const char* prog_name)
{
	printf("Usage: %s [NDIM] [NGPU] [NCPU] [-pvh]\n", prog_name );
	printf("\n");
	printf("OPTIONS\n");
	printf("  -p : print matrix data.\n");
	printf("  -v : validate matrix multiplication.\n");
	printf("  -h : print this page.\n");
}

void parse_opt(int argc, char** argv)
{
	int opt;

	while( (opt = getopt(argc, argv, "pvhikjs:")) != -1 )
	{
		switch(opt)
		{
		case 'p':
			// print matrix data.
			print_matrix = 1;
			break;

		case 'v':
			// validation
			validation = 1;
			break;

		case 'h':
		default:
			print_help(argv[0]);
			exit(0);
			break;
		}
	}
}

int main(int argc, char** argv)
{

  int i, j, k = 1;
  int NDIM = 1024;
  int NGPU = 1;
  int NCPU = 0;
  float * a, * b, * c;

  NDIM = atoi(argv[1]);
  NGPU = atoi(argv[2]);
  NCPU = atoi(argv[3]);
	parse_opt( argc, argv );

  a = (float *)malloc(sizeof(float) * NDIM * NDIM);
  b = (float *)malloc(sizeof(float) * NDIM * NDIM);
  c = (float *)malloc(sizeof(float) * NDIM * NDIM);

  printf("%d x %d x %d\n", NDIM, NDIM, NDIM);
  printf("GPU %d, CPU %d\n", NGPU, NCPU);

	for( i = 0; i < NDIM; i++ )
	{
		for( j = 0; j < NDIM; j++ )
		{
			a[i * NDIM + j] = k;
			b[i * NDIM + j] = k;
			k++;
		}
	}

	timer_start(1);
	mat_mul( c, a, b, NDIM, NGPU, NCPU );
	timer_stop(1);

	printf("Time elapsed : %lf sec\n", timer_read(1));

	if( validation )
		check_mat_mul( c, a, b, NDIM );

	if( print_matrix )
	{
		printf("MATRIX A: \n");
		print_mat(a, NDIM);

		printf("MATRIX B: \n");
		print_mat(b, NDIM);

		printf("MATRIX C: \n");
		print_mat(c, NDIM);
	}

	return 0;
	//cl_device_id *device = malloc(sizeof(cl_device_id) * device_num);
}
