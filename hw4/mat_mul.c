#include <CL/cl.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include "timers.h"

int print_matrix = 0;
int validation = 0;

const char* kernel_src = 
				"__kernel void vec_mul(__global float *A, " \
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

void mat_mul( float *c, float *a, float *b, int NDIM) {
	cl_int error;

	cl_platform_id platform;
	clGetPlatformIDs(1, &platform, NULL);

	cl_device_id device;
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

	cl_context context;
	context = clCreateContext(0, 1, &device, NULL, NULL, NULL);

	cl_command_queue command_queue;
	command_queue = clCreateCommandQueue(context, device, 0, NULL);

	cl_mem bufferA;
	cl_mem bufferB;
	cl_mem bufferC;

	size_t sizeA = NDIM * NDIM * sizeof(float);
	size_t sizeB = NDIM * NDIM * sizeof(float);
	size_t sizeC = NDIM * NDIM * sizeof(float);

	bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeA, NULL, NULL);
	bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeB, NULL, NULL);
	bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeC, NULL, NULL);

	int NDIM_p = NDIM;

	cl_program program;
	size_t kernel_src_len = strlen(kernel_src);
	program = clCreateProgramWithSource(context, 1, (const char**) &kernel_src, &kernel_src_len, NULL);

	error = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	if (error != CL_SUCCESS) {
		printf("Program build fail\n");
		size_t len;
		char buffer[2048];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);

		printf("%s\n", buffer);
		exit(1);
	}

	cl_kernel kernel;
	kernel = clCreateKernel(program, "vec_mul", NULL);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &bufferA);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &bufferB);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &bufferC);
	clSetKernelArg(kernel, 3, sizeof(int), (void *)&NDIM_p);

	clEnqueueWriteBuffer(command_queue, bufferA, CL_FALSE, 0, sizeA, a, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, bufferB, CL_FALSE, 0, sizeB, b, 0, NULL, NULL);

	size_t global[2] = {NDIM, NDIM};
	size_t local[2] = {16, 16};
	clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global, local, 0, NULL, NULL);

	clFinish(command_queue);

	clEnqueueReadBuffer(command_queue, bufferC, CL_TRUE, 0, sizeC, c, 0, NULL, NULL);
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
	printf("Usage: %s [NDIM] [-pvh]\n", prog_name );
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
  float * a, * b, * c;

  NDIM = atoi(argv[1]);
	parse_opt( argc, argv );

  a = (float *)malloc(sizeof(float) * NDIM * NDIM);
  b = (float *)malloc(sizeof(float) * NDIM * NDIM);
  c = (float *)malloc(sizeof(float) * NDIM * NDIM);

  printf("%d x %d x %d\n", NDIM, NDIM, NDIM);

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
	mat_mul( c, a, b, NDIM );
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
}
