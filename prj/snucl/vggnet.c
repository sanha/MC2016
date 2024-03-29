#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void build_program_check(cl_int error, cl_program program, cl_device_id *device);
void create_kernel_error_check(cl_int error, char *where);
void enqueue_kernel_error_check(cl_int error, char *where);
void enqueue_buffer_error_check(cl_int error, char *where, char *what);
void conv_single_itr_test(float *input_tmp, float *output_tmp, float *filter_tmp, int N);

const char* pooling_kernel_src =
                "__kernel void pooling(__global const float *inputs, " \
                "__global float *outputs, " \
                "int N) " \
                "{" \
                "   int g_id = get_global_id(0);" \
                "   int i, j, k, l;" \
                "   int input_size = 4 * N * N;" \
                "   int output_size = N * N;" \
                "   __global const float *input = inputs + g_id * input_size;" \
                "   __global float *output = outputs + g_id * output_size;" \
                "   float max;" \
                "   int offset;" \
                "   for (i = 0; i < N; i++) {" \
                "       for (j = 0; j < N; j++) {" \
                "           max = 0.0;" \
                "           for (k = 0; k < 2; k++) { " \
                "               offset = (i * 2 + k) * 2 * N + j * 2;" \
                "               for (l = 0; l < 2; l++) { " \
                "                   float pixel = input[offset + l]; " \
                "                   max = (max > pixel) ? max : pixel; " \
                "               }" \
                "           }" \
                "           output[i * N + j] = max;" \
                "       }" \
                "   }" \
                "}";

const char* mat_conversion_kernel_src =
				"__kernel void mat_conversion(__global const float* inputs, " \
				"__global float* mats) {" \
				"	int g_1 = get_global_id(0);" \
				"	int g_2 = get_global_id(1);" \
				"	int g_3 = get_global_id(2);" \
				"	int N = get_global_size(1);" \
				"	int nn = N * N;" \
				" 	int i, j;" \
				"	__global const float *input = inputs + g_1 * nn;" \
				"	__global float *mat = mats + g_1 * 9 * nn;" \
				"	int idx;" \
				"   for (i = 0; i < 3; i++) {" \
				"       int x = g_2 + i - 1;" \
				"   	for (j = 0; j < 3; j++) {" \
				"       	int y = g_3 + j - 1;" \
				"			idx = 3 * i + j;" \
				"       	float val = 0.0;" \
				"       	if (x >= 0 && x < N && y >= 0 && y < N) {" \
				"       		val = input[x * N + y];" \
				"   		}" \
				"			mat[idx * nn + g_2 * N + g_3] = val;" \
				"		}" \
				"	}" \
				"}";

const char* conv_kernel_src =
                "__kernel void convolution(__global const float *mats, " \
                "__global float *outputs, " \
                "__global const float *filters, " \
				"__global const float *biases, " \
                "int D1)" \
                "{" \
                "   int g_1 = get_global_id(0);" \
                "   int D2 = get_global_size(0);" \
				"	int g_2 = get_global_id(1);" \
				"	int g_3 = get_global_id(2);" \
				"	int N = get_global_size(1);" \
				"	int nn = N * N;" \
				"	__global const float *filter = filters + g_1 * 9 * D1;" \
				"	float sum = 0.0;" \
				"	int i;" \
				"	for (i = 0; i < 9 * D1; i++) {" \
				"		sum += filter[i] * mats[i * nn + g_2 * N + g_3]; " \
				"	}" \
				"	sum += biases[g_1];" \
				"	outputs[g_1 * nn + g_2 * N + g_3] = sum > 0 ? sum : 0;" \
               "}";

const char* fc_kernel_src =
                "__kernel void fc(__global float *input_neuron, " \
                "__global float *output_neuron, " \
                "__global float *weights, " \
                "__global float *biases, " \
                "int N) " \
                "{" \
                "   int g_id = get_global_id(0);" \
                "   int i;" \
                "   float sum = 0;" \
                "   int offset = g_id * N;" \
                "   for (i = 0; i < N; i++) {" \
                "       sum += input_neuron[i] * weights[offset + i];" \
                "   }" \
                "   sum += biases[g_id];" \
                "   output_neuron[g_id] = sum > 0 ? sum : 0;" \
                "}";

cl_platform_id *platform;
cl_device_id *device;
cl_context *context;
cl_command_queue *command_queue;
cl_program *pooling_program, *mat_conversion_program, *conv_program, *fc_program;
cl_mem *inout1_buf, *inout2_buf, *mats_buf;
cl_mem *f1_1_buf, *f1_2_buf, *f2_1_buf, *f2_2_buf, *f3_1_buf, *f3_2_buf, *f3_3_buf, *f4_1_buf, *f4_2_buf, *f4_3_buf, *f5_1_buf, *f5_2_buf, *f5_3_buf;
cl_mem *b1_1_buf, *b1_2_buf, *b2_1_buf, *b2_2_buf, *b3_1_buf, *b3_2_buf, *b3_3_buf, *b4_1_buf, *b4_2_buf, *b4_3_buf, *b5_1_buf, *b5_2_buf, *b5_3_buf;
float *f1_1, *f1_2, *f2_1, *f2_2, *f3_1, *f3_2, *f3_3, *f4_1, *f4_2, *f4_3, *f5_1, *f5_2, *f5_3;
float *b1_1, *b1_2, *b2_1, *b2_2, *b3_1, *b3_2, *b3_3, *b4_1, *b4_2, *b4_3, *b5_1, *b5_2, *b5_3;
float *w1, *w2, *w3;
float *b1, *b2, *b3;

static float * get_param(float ** array, int size)
{
  float * subarray = *array;
  *array += size;
  return subarray;
}

int cl_setting(float *network) {
  platform = (cl_platform_id *)malloc(sizeof(platform));
  device = (cl_device_id *)malloc(sizeof(cl_device_id) * 16);
  context = (cl_context *)malloc(sizeof(cl_device_id));
  command_queue = (cl_command_queue *)malloc(sizeof(cl_command_queue) * 16);
  pooling_program = (cl_program *)malloc(sizeof(cl_program));
  mat_conversion_program = (cl_program *)malloc(sizeof(cl_program));
  conv_program = (cl_program *)malloc(sizeof(cl_program));
  fc_program = (cl_program *)malloc(sizeof(cl_program));
  inout1_buf = (cl_mem *)malloc(sizeof(cl_mem) * 16);
  inout2_buf = (cl_mem *)malloc(sizeof(cl_mem) * 16);
  mats_buf = (cl_mem *)malloc(sizeof(cl_mem) * 16);
  f1_1_buf = (cl_mem *)malloc(sizeof(cl_mem) * 16);
  f1_2_buf = (cl_mem *)malloc(sizeof(cl_mem) * 16);
  f2_1_buf = (cl_mem *)malloc(sizeof(cl_mem) * 16);
  f2_2_buf = (cl_mem *)malloc(sizeof(cl_mem) * 16);
  f3_1_buf = (cl_mem *)malloc(sizeof(cl_mem) * 16);
  f3_2_buf = (cl_mem *)malloc(sizeof(cl_mem) * 16);
  f3_3_buf = (cl_mem *)malloc(sizeof(cl_mem) * 16);
  f4_1_buf = (cl_mem *)malloc(sizeof(cl_mem) * 16);
  f4_2_buf = (cl_mem *)malloc(sizeof(cl_mem) * 16);
  f4_3_buf = (cl_mem *)malloc(sizeof(cl_mem) * 16);
  f5_1_buf = (cl_mem *)malloc(sizeof(cl_mem) * 16);
  f5_2_buf = (cl_mem *)malloc(sizeof(cl_mem) * 16);
  f5_3_buf = (cl_mem *)malloc(sizeof(cl_mem) * 16);
  b1_1_buf = (cl_mem *)malloc(sizeof(cl_mem) * 16);
  b1_2_buf = (cl_mem *)malloc(sizeof(cl_mem) * 16);
  b2_1_buf = (cl_mem *)malloc(sizeof(cl_mem) * 16);
  b2_2_buf = (cl_mem *)malloc(sizeof(cl_mem) * 16);
  b3_1_buf = (cl_mem *)malloc(sizeof(cl_mem) * 16);
  b3_2_buf = (cl_mem *)malloc(sizeof(cl_mem) * 16);
  b3_3_buf = (cl_mem *)malloc(sizeof(cl_mem) * 16);
  b4_1_buf = (cl_mem *)malloc(sizeof(cl_mem) * 16);
  b4_2_buf = (cl_mem *)malloc(sizeof(cl_mem) * 16);
  b4_3_buf = (cl_mem *)malloc(sizeof(cl_mem) * 16);
  b5_1_buf = (cl_mem *)malloc(sizeof(cl_mem) * 16);
  b5_2_buf = (cl_mem *)malloc(sizeof(cl_mem) * 16);
  b5_3_buf = (cl_mem *)malloc(sizeof(cl_mem) * 16);
  
  cl_int error;
  int ndev, i;

  error = clGetPlatformIDs(1, platform, NULL);
  if (error != CL_SUCCESS) {
    printf("Get platform id fail\n");
    exit(1);
  }

  error = clGetDeviceIDs(*platform, CL_DEVICE_TYPE_GPU, 0, NULL, (unsigned int *) &ndev);
  if (error != CL_SUCCESS) {
    printf("Get cpu number fail\n");
    exit(1);
  } else if (ndev != 16) {
  	printf("gpu device number is %d instead of 16\n", ndev);
	exit(1);
  }
 
  error = clGetDeviceIDs(*platform, CL_DEVICE_TYPE_GPU, ndev, device, NULL);
  if (error != CL_SUCCESS) {
    printf("Get gpu id fail\n");
    exit (1);
  }

  *context = clCreateContext(0, ndev, device, NULL, NULL, &error);
  if (error != CL_SUCCESS) {
    printf("Context creation failed\n");
    exit(1);
  }

  for (i = 0; i < ndev; i++) {
    command_queue[i] = clCreateCommandQueue(*context, device[i], CL_QUEUE_PROFILING_ENABLE, &error);
    if (error != CL_SUCCESS) {
      printf("%d'th command queue creation failed\n", i);
    }
  }

  size_t pooling_kernel_src_len = strlen(pooling_kernel_src);
  *pooling_program = clCreateProgramWithSource(*context, 1, (const char **) &pooling_kernel_src, &pooling_kernel_src_len, NULL);
  error = clBuildProgram(*pooling_program, ndev, device, NULL, NULL, NULL);
  build_program_check(error, *pooling_program, device);

  size_t mat_conversion_kernel_src_len = strlen(mat_conversion_kernel_src);
  *mat_conversion_program = clCreateProgramWithSource(*context, 1, (const char **) &mat_conversion_kernel_src, &mat_conversion_kernel_src_len, NULL);
  error = clBuildProgram(*mat_conversion_program, ndev, device, NULL, NULL, NULL);
  build_program_check(error, *mat_conversion_program, device);

  size_t conv_kernel_src_len = strlen(conv_kernel_src);
  *conv_program = clCreateProgramWithSource(*context, 1, (const char **) &conv_kernel_src, &conv_kernel_src_len, NULL);
  error = clBuildProgram(*conv_program, ndev, device, NULL, NULL, NULL);
  build_program_check(error, *conv_program, device);

  size_t fc_kernel_src_len = strlen(fc_kernel_src);
  *fc_program = clCreateProgramWithSource(*context, 1, (const char **) &fc_kernel_src, &fc_kernel_src_len, NULL);
  error = clBuildProgram(*fc_program, ndev, device, NULL, NULL, NULL);
  build_program_check(error, *fc_program, device);

  f1_1 = get_param(&network, 3 * 3 * 3 * 64);
  b1_1 = get_param(&network, 64);
  f1_2 = get_param(&network, 3 * 3 * 64 * 64);
  b1_2 = get_param(&network, 64);

  f2_1 = get_param(&network, 3 * 3 * 64 * 128);
  b2_1 = get_param(&network, 128);
  f2_2 = get_param(&network, 3 * 3 * 128 * 128);
  b2_2 = get_param(&network, 128);

  f3_1 = get_param(&network, 3 * 3 * 128 * 256);
  b3_1 = get_param(&network, 256);
  f3_2 = get_param(&network, 3 * 3 * 256 * 256);
  b3_2 = get_param(&network, 256);
  f3_3 = get_param(&network, 3 * 3 * 256 * 256);
  b3_3 = get_param(&network, 256);

  f4_1 = get_param(&network, 3 * 3 * 256 * 512);
  b4_1 = get_param(&network, 512);
  f4_2 = get_param(&network, 3 * 3 * 512 * 512);
  b4_2 = get_param(&network, 512);
  f4_3 = get_param(&network, 3 * 3 * 512 * 512);
  b4_3 = get_param(&network, 512);

  f5_1 = get_param(&network, 3 * 3 * 512 * 512);
  b5_1 = get_param(&network, 512);
  f5_2 = get_param(&network, 3 * 3 * 512 * 512);
  b5_2 = get_param(&network, 512);
  f5_3 = get_param(&network, 3 * 3 * 512 * 512);
  b5_3 = get_param(&network, 512);

  w1 = get_param(&network, 7 * 7 * 512 * 4096);
  b1 = get_param(&network, 4096);
  w2 = get_param(&network, 4096 * 4096);
  b2 = get_param(&network, 4096);
  w3 = get_param(&network, 4096 * 1000);
  b3 = get_param(&network, 1000);

  for (i = 0; i < ndev; i++) {
    inout1_buf[i] = clCreateBuffer(*context, CL_MEM_READ_WRITE, 224 * 224 * 64 * sizeof(float), NULL, NULL);
    inout2_buf[i] = clCreateBuffer(*context, CL_MEM_READ_WRITE, 224 * 224 * 64 * sizeof(float), NULL, NULL);
    mats_buf[i] = clCreateBuffer(*context, CL_MEM_READ_WRITE, 224 * 224 * 64 * 9 * sizeof(float), NULL, NULL);

    f1_1_buf[i] = clCreateBuffer(*context, CL_MEM_READ_ONLY, 3 * 3 * 3 * 64 * sizeof(float), NULL, NULL);
    b1_1_buf[i] = clCreateBuffer(*context, CL_MEM_READ_ONLY, 64 * sizeof(float), NULL, NULL);
    f1_2_buf[i] = clCreateBuffer(*context, CL_MEM_READ_ONLY, 3 * 3 * 64 * 64 * sizeof(float), NULL, NULL);
    b1_2_buf[i] = clCreateBuffer(*context, CL_MEM_READ_ONLY, 64 * sizeof(float), NULL, NULL);

    error = clEnqueueWriteBuffer(command_queue[i], f1_1_buf[i], CL_FALSE, 0, 3 * 3 * 3 * 64 * sizeof(float), (void *) ((size_t) f1_1), 0, NULL, NULL);
    enqueue_buffer_error_check(error, "cl_setting", "f1_1");
    error = clEnqueueWriteBuffer(command_queue[i], b1_1_buf[i], CL_FALSE, 0, 64 * sizeof(float), (void *) ((size_t) b1_1), 0, NULL, NULL);
    enqueue_buffer_error_check(error, "cl_setting", "b1_1");
    error = clEnqueueWriteBuffer(command_queue[i], f1_2_buf[i], CL_FALSE, 0, 3 * 3 * 64 * 64 * sizeof(float), (void *) ((size_t) f1_2), 0, NULL, NULL);
    enqueue_buffer_error_check(error, "cl_setting", "f1_2");
    error = clEnqueueWriteBuffer(command_queue[i], b1_2_buf[i], CL_FALSE, 0, 64 * sizeof(float), (void *) ((size_t) b1_2), 0, NULL, NULL);
    enqueue_buffer_error_check(error, "cl_setting", "b1_2");

    f2_1_buf[i] = clCreateBuffer(*context, CL_MEM_READ_ONLY, 3 * 3 * 64 * 128 * sizeof(float), NULL, NULL);
    b2_1_buf[i] = clCreateBuffer(*context, CL_MEM_READ_ONLY, 128 * sizeof(float), NULL, NULL);
    f2_2_buf[i] = clCreateBuffer(*context, CL_MEM_READ_ONLY, 3 * 3 * 128 * 128 * sizeof(float), NULL, NULL);
    b2_2_buf[i] = clCreateBuffer(*context, CL_MEM_READ_ONLY, 128 * sizeof(float), NULL, NULL);

    error = clEnqueueWriteBuffer(command_queue[i], f2_1_buf[i], CL_FALSE, 0, 3 * 3 * 64 * 128 * sizeof(float), (void *) ((size_t) f2_1), 0, NULL, NULL);
    enqueue_buffer_error_check(error, "cl_setting", "f2_1");
    error = clEnqueueWriteBuffer(command_queue[i], b2_1_buf[i], CL_FALSE, 0, 128 * sizeof(float), (void *) ((size_t) b2_1), 0, NULL, NULL);
    enqueue_buffer_error_check(error, "cl_setting", "b2_1");
    error = clEnqueueWriteBuffer(command_queue[i], f2_2_buf[i], CL_FALSE, 0, 3 * 3 * 128 * 128 * sizeof(float), (void *) ((size_t) f2_2), 0, NULL, NULL);
    enqueue_buffer_error_check(error, "cl_setting", "f2_2");
    error = clEnqueueWriteBuffer(command_queue[i], b2_2_buf[i], CL_FALSE, 0, 128 * sizeof(float), (void *) ((size_t) b2_2), 0, NULL, NULL);
    enqueue_buffer_error_check(error, "cl_setting", "b2_2");

    f3_1_buf[i] = clCreateBuffer(*context, CL_MEM_READ_ONLY, 3 * 3 * 128 * 256 * sizeof(float), NULL, NULL);
    b3_1_buf[i] = clCreateBuffer(*context, CL_MEM_READ_ONLY, 256 * sizeof(float), NULL, NULL);
    f3_2_buf[i] = clCreateBuffer(*context, CL_MEM_READ_ONLY, 3 * 3 * 256 * 256 * sizeof(float), NULL, NULL);
    b3_2_buf[i] = clCreateBuffer(*context, CL_MEM_READ_ONLY, 256 * sizeof(float), NULL, NULL);
    f3_3_buf[i] = clCreateBuffer(*context, CL_MEM_READ_ONLY, 3 * 3 * 256 * 256 * sizeof(float), NULL, NULL);
    b3_3_buf[i] = clCreateBuffer(*context, CL_MEM_READ_ONLY, 256 * sizeof(float), NULL, NULL);

    error = clEnqueueWriteBuffer(command_queue[i], f3_1_buf[i], CL_FALSE, 0, 3 * 3 * 128 * 256 * sizeof(float), (void *) ((size_t) f3_1), 0, NULL, NULL);
    enqueue_buffer_error_check(error, "cl_setting", "f3_1");
    error = clEnqueueWriteBuffer(command_queue[i], b3_1_buf[i], CL_FALSE, 0, 256 * sizeof(float), (void *) ((size_t) b3_1), 0, NULL, NULL);
    enqueue_buffer_error_check(error, "cl_setting", "b3_1");
    error = clEnqueueWriteBuffer(command_queue[i], f3_2_buf[i], CL_FALSE, 0, 3 * 3 * 256 * 256 * sizeof(float), (void *) ((size_t) f3_2), 0, NULL, NULL);
    enqueue_buffer_error_check(error, "cl_setting", "f3_2");
    error = clEnqueueWriteBuffer(command_queue[i], b3_2_buf[i], CL_FALSE, 0, 256 * sizeof(float), (void *) ((size_t) b3_2), 0, NULL, NULL);
    enqueue_buffer_error_check(error, "cl_setting", "b3_2");
    error = clEnqueueWriteBuffer(command_queue[i], f3_3_buf[i], CL_FALSE, 0, 3 * 3 * 256 * 256 * sizeof(float), (void *) ((size_t) f3_3), 0, NULL, NULL);
    enqueue_buffer_error_check(error, "cl_setting", "f3_3");
    error = clEnqueueWriteBuffer(command_queue[i], b3_3_buf[i], CL_FALSE, 0, 256 * sizeof(float), (void *) ((size_t) b3_3), 0, NULL, NULL);
    enqueue_buffer_error_check(error, "cl_setting", "b3_3");

    f4_1_buf[i] = clCreateBuffer(*context, CL_MEM_READ_ONLY, 3 * 3 * 256 * 512 * sizeof(float), NULL, NULL);
    b4_1_buf[i] = clCreateBuffer(*context, CL_MEM_READ_ONLY, 512 * sizeof(float), NULL, NULL);
    f4_2_buf[i] = clCreateBuffer(*context, CL_MEM_READ_ONLY, 3 * 3 * 512 * 512 * sizeof(float), NULL, NULL);
    b4_2_buf[i] = clCreateBuffer(*context, CL_MEM_READ_ONLY, 512 * sizeof(float), NULL, NULL);
    f4_3_buf[i] = clCreateBuffer(*context, CL_MEM_READ_ONLY, 3 * 3 * 512 * 512 * sizeof(float), NULL, NULL);
    b4_3_buf[i] = clCreateBuffer(*context, CL_MEM_READ_ONLY, 512 * sizeof(float), NULL, NULL);

    error = clEnqueueWriteBuffer(command_queue[i], f4_1_buf[i], CL_FALSE, 0, 3 * 3 * 256 * 512 * sizeof(float), (void *) ((size_t) f4_1), 0, NULL, NULL);
    enqueue_buffer_error_check(error, "cl_setting", "f4_1");
    error = clEnqueueWriteBuffer(command_queue[i], b4_1_buf[i], CL_FALSE, 0, 512 * sizeof(float), (void *) ((size_t) b4_1), 0, NULL, NULL);
    enqueue_buffer_error_check(error, "cl_setting", "b4_1");
    error = clEnqueueWriteBuffer(command_queue[i], f4_2_buf[i], CL_FALSE, 0, 3 * 3 * 512 * 512 * sizeof(float), (void *) ((size_t) f4_2), 0, NULL, NULL);
    enqueue_buffer_error_check(error, "cl_setting", "f4_2");
    error = clEnqueueWriteBuffer(command_queue[i], b4_2_buf[i], CL_FALSE, 0, 512 * sizeof(float), (void *) ((size_t) b4_2), 0, NULL, NULL);
    enqueue_buffer_error_check(error, "cl_setting", "b4_2");
    error = clEnqueueWriteBuffer(command_queue[i], f4_3_buf[i], CL_FALSE, 0, 3 * 3 * 512 * 512 * sizeof(float), (void *) ((size_t) f4_3), 0, NULL, NULL);
    enqueue_buffer_error_check(error, "cl_setting", "f4_3");
    error = clEnqueueWriteBuffer(command_queue[i], b4_3_buf[i], CL_FALSE, 0, 512 * sizeof(float), (void *) ((size_t) b4_3), 0, NULL, NULL);
    enqueue_buffer_error_check(error, "cl_setting", "b4_3");

    f5_1_buf[i] = clCreateBuffer(*context, CL_MEM_READ_ONLY, 3 * 3 * 512 * 512 * sizeof(float), NULL, NULL);
    b5_1_buf[i] = clCreateBuffer(*context, CL_MEM_READ_ONLY, 512 * sizeof(float), NULL, NULL);
    f5_2_buf[i] = clCreateBuffer(*context, CL_MEM_READ_ONLY, 3 * 3 * 512 * 512 * sizeof(float), NULL, NULL);
    b5_2_buf[i] = clCreateBuffer(*context, CL_MEM_READ_ONLY, 512 * sizeof(float), NULL, NULL);
    f5_3_buf[i] = clCreateBuffer(*context, CL_MEM_READ_ONLY, 3 * 3 * 512 * 512 * sizeof(float), NULL, NULL);
    b5_3_buf[i] = clCreateBuffer(*context, CL_MEM_READ_ONLY, 512 * sizeof(float), NULL, NULL);

    error = clEnqueueWriteBuffer(command_queue[i], f5_1_buf[i], CL_FALSE, 0, 3 * 3 * 512 * 512 * sizeof(float), (void *) ((size_t) f5_1), 0, NULL, NULL);
    enqueue_buffer_error_check(error, "cl_setting", "f5_1");
    error = clEnqueueWriteBuffer(command_queue[i], b5_1_buf[i], CL_FALSE, 0, 512 * sizeof(float), (void *) ((size_t) b5_1), 0, NULL, NULL);
    enqueue_buffer_error_check(error, "cl_setting", "b5_1");
    error = clEnqueueWriteBuffer(command_queue[i], f5_2_buf[i], CL_FALSE, 0, 3 * 3 * 512 * 512 * sizeof(float), (void *) ((size_t) f5_2), 0, NULL, NULL);
    enqueue_buffer_error_check(error, "cl_setting", "f5_2");
    error = clEnqueueWriteBuffer(command_queue[i], b5_2_buf[i], CL_FALSE, 0, 512 * sizeof(float), (void *) ((size_t) b5_2), 0, NULL, NULL);
    enqueue_buffer_error_check(error, "cl_setting", "b5_2");
    error = clEnqueueWriteBuffer(command_queue[i], f5_3_buf[i], CL_FALSE, 0, 3 * 3 * 512 * 512 * sizeof(float), (void *) ((size_t) f5_3), 0, NULL, NULL);
    enqueue_buffer_error_check(error, "cl_setting", "f5_3");
    error = clEnqueueWriteBuffer(command_queue[i], b5_3_buf[i], CL_FALSE, 0, 512 * sizeof(float), (void *) ((size_t) b5_3), 0, NULL, NULL);
    enqueue_buffer_error_check(error, "cl_setting", "b5_3");
  }
  
  return ndev;
}

static void pooling2x2(float * input, float * output, int N)
{ // N: output demension
  int i, j, k, l;
  for(i = 0; i < N; i++)
  {
    for(j = 0; j < N; j++)
    {
      float max = 0;
      for(k = 0; k < 2; k++)
      {
        for(l = 0; l < 2; l++)
        {
          float pixel = input[(i * 2 + k) * 2 * N + j * 2 + l];
          max = (max > pixel) ? max : pixel;
        }
      }
      output[i * N + j] = max;
    }
  }
}

static void pooling_layer_buf(cl_mem *inputs_bufs, cl_mem *outputs_bufs, int N, int D, int ndev)
{
  cl_int error;
  int n = N;
  size_t global[1] = { D };
  size_t local[1] = { 64 };
  cl_kernel kernel[ndev];
  int i;

  for (i = 0; i < ndev; i ++) {
    kernel[i] = clCreateKernel(*pooling_program, "pooling", &error);
    create_kernel_error_check(error, "pooling");

    clSetKernelArg(kernel[i], 0, sizeof(cl_mem), (void *)&inputs_bufs[i]);
    clSetKernelArg(kernel[i], 1, sizeof(cl_mem), (void *)&outputs_bufs[i]);
    clSetKernelArg(kernel[i], 2, sizeof(int), (void *)&n);

    error = clEnqueueNDRangeKernel(command_queue[i], kernel[i], 1, NULL, global, local, 0, NULL, NULL);
    enqueue_kernel_error_check(error, "pooling");
  }
}

static void pooling_layer_d(float * inputs, float * outputs, int N, int D)
{
  int i;
  for(i = 0; i < D; i++)
  {
    float * input = inputs + i * N * N * 4;
    float * output = outputs + i * N * N;
    pooling2x2(input, output, N);
  }
}

static void convolution3x3(float * input, float * output, float * filter, int N)
{
  int i, j, k, l;
  for(i = 0; i < N; i++)
  {
    for(j = 0; j < N; j++)
    {
      float sum = 0;
      for(k = 0; k < 3; k++)
      {
        int x = i + k - 1;
        for(l = 0; l < 3; l++)
        {
          int y = j + l - 1; 
          if(x >= 0 && x < N && y >= 0 && y < N)
            sum += input[x * N + y] * filter[k * 3 + l];
        }
      }
      output[i * N + j] += sum;
    }
  }
}

#define ReLU(x) (((x)>0)?(x):0)
static void convolution_layer_buf(cl_mem *inputs_bufs, cl_mem *outputs_bufs, cl_mem *mats_buf, cl_mem *filters_bufs, cl_mem *biases_bufs, int N, int D1, int D2, int ndev)
{
  cl_int error;
//  int inputs_size = D1 * N * N * sizeof(float);
//  int mats_size = 9 * inputs_size;
  int d1 = D1;
  size_t global[3] = { D1, N, N };
  size_t local[3] = { 1, 1, N };

//  cl_mem mats_buf[ndev];
  cl_kernel kernel[ndev];

  int i;
  for (i = 0; i < ndev; i++) {
//    mats_buf[i] = clCreateBuffer(*context, CL_MEM_READ_WRITE, mats_size, NULL, NULL);

    kernel[i] = clCreateKernel(*mat_conversion_program, "mat_conversion", &error);
    create_kernel_error_check(error, "mat_conversion");

    clSetKernelArg(kernel[i], 0, sizeof(cl_mem), (void *)&inputs_bufs[i]);
    clSetKernelArg(kernel[i], 1, sizeof(cl_mem), (void *)&mats_buf[i]);
    
	error = clEnqueueNDRangeKernel(command_queue[i], kernel[i], 3, NULL, global, local, 0, NULL, NULL);
    enqueue_kernel_error_check(error, "mat_conversion");
  }

  global[0] = D2;
  global[1] = N;
  global[2] = N;

  int local_size;
  local[0] = 4;
  if (!(N % 8)) {
    local_size = 8;
  } else if (!(N % 4)) {
    local_size = 4;
  } else {
    local_size = 2;
  }
  local[1] = local_size;
  local[2] = local_size;

  for (i = 0; i < ndev; i++) {
    kernel[i] = clCreateKernel(*conv_program, "convolution", &error);
    create_kernel_error_check(error, "convolution");

    clSetKernelArg(kernel[i], 0, sizeof(cl_mem), (void *)&mats_buf[i]);
    clSetKernelArg(kernel[i], 1, sizeof(cl_mem), (void *)&outputs_bufs[i]);
    clSetKernelArg(kernel[i], 2, sizeof(cl_mem), (void *)&filters_bufs[i]);
    clSetKernelArg(kernel[i], 3, sizeof(cl_mem), (void *)&biases_bufs[i]);
    clSetKernelArg(kernel[i], 4, sizeof(int), (void *)&d1);

    error = clEnqueueNDRangeKernel(command_queue[i], kernel[i], 3, NULL, global, local, 0, NULL, NULL);
    enqueue_kernel_error_check(error, "convolution");
  }
}

static void convolution_layer_d(float * inputs, float * outputs, float * filters, float * biases, int N, int D1, int D2)
{
  int i, j;

  memset(outputs, 0, sizeof(float) * N * N * D2);

  for(j = 0; j < D2; j++)
  {
    float * output = outputs + N * N * j;
    for(i = 0; i < D1; i++)
    {
      float * input = inputs + N * N * i;
      float * filter = filters + 3 * 3 * (j * D1 + i);
      convolution3x3(input, output, filter, N);
    }
  }

  for(i = 0; i < D2; i++)
  {
    float * output = outputs + N * N * i;
    float bias = biases[i];
    for(j = 0; j < N * N; j++)
    {
      output[j] = ReLU(output[j] + bias);
    }
  }
}

static void fc_layer(float **input_neurons, float **output_neurons, float * weights, float * biases, int N, int M, int ndev) 
{
  cl_int error;
  int input_n_size = N * sizeof(float);
  int output_n_size = M * sizeof(float);
  int weights_size = N * M * sizeof(float);
  int biases_size = M * sizeof(float);
  int n = N;
  size_t global[1] = { M };
  size_t local[1] = { 64 };

  cl_kernel kernel[ndev];
  cl_mem input_n_buf[ndev];
  cl_mem output_n_buf[ndev];
  cl_mem weights_buf[ndev];
  cl_mem biases_buf[ndev];

  int i;
  for (i = 0; i < ndev; i++) {
    input_n_buf[i] = clCreateBuffer(*context, CL_MEM_READ_ONLY, input_n_size, NULL, NULL);
    output_n_buf[i] = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, output_n_size, NULL, NULL);
    weights_buf[i] = clCreateBuffer(*context, CL_MEM_READ_ONLY, weights_size, NULL, NULL);
    biases_buf[i] = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, biases_size, NULL, NULL);

    kernel[i] = clCreateKernel(*fc_program, "fc", &error);
    create_kernel_error_check(error, "fc");

    clSetKernelArg(kernel[i], 0, sizeof(cl_mem), (void *)&input_n_buf[i]);
    clSetKernelArg(kernel[i], 1, sizeof(cl_mem), (void *)&output_n_buf[i]);
    clSetKernelArg(kernel[i], 2, sizeof(cl_mem), (void *)&weights_buf[i]);
    clSetKernelArg(kernel[i], 3, sizeof(cl_mem), (void *)&biases_buf[i]);
    clSetKernelArg(kernel[i], 4, sizeof(int), (void *)&n);

    error = clEnqueueWriteBuffer(command_queue[i], input_n_buf[i], CL_FALSE, 0, input_n_size, (void *) ((size_t) input_neurons[i]), 0, NULL, NULL);
    enqueue_buffer_error_check(error, "fc", "input_neuron");
    error = clEnqueueWriteBuffer(command_queue[i], weights_buf[i], CL_FALSE, 0, weights_size, (void *) ((size_t) weights), 0, NULL, NULL);
    enqueue_buffer_error_check(error, "fc", "weights");
    error = clEnqueueWriteBuffer(command_queue[i], biases_buf[i], CL_FALSE, 0, biases_size, (void *) ((size_t) biases), 0, NULL, NULL);
    enqueue_buffer_error_check(error, "fc", "biases");

    error = clEnqueueNDRangeKernel(command_queue[i], kernel[i], 1, NULL, global, local, 0, NULL, NULL);
    enqueue_kernel_error_check(error, "fc");
  }
  for (i = 0; i < ndev; i++) {
    error = clEnqueueReadBuffer(command_queue[i], output_n_buf[i], CL_TRUE, 0, output_n_size, (void *) ((size_t) output_neurons[i]), 0, NULL, NULL);
    enqueue_buffer_error_check(error, "fc", "output_neuron");
  }
}

static void fc_layer_d(float * input_neuron, float * output_neuron, float * weights, float * biases, int N, int M)
{
  int i, j;
  for(j = 0; j < M; j++)
  {
    float sum = 0;
    for(i = 0; i < N; i++)
    {
      sum += input_neuron[i] * weights[j * N + i];
    }
    sum += biases[j];
    output_neuron[j] = ReLU(sum);
  }
}

static void softmax(float * output)
{
  int i;
  float max = output[0];
  for(i = 1; i < 1000; i++)
  {
    max = (output[i] > max)?output[i]:max;
  }
  float sum = 0;
  for(i = 0; i < 1000; i++)
  {
    sum += exp(output[i] - max);
  }
  for(i = 0; i < 1000; i++)
  {
    output[i] = exp(output[i] - max) / sum;
  }
}

static int find_max(float * fc)
{
  int i;
  int maxid = 0;
  float maxval = 0;
  for(i = 0; i < 1000; i++)
  {
    if(maxval < fc[i])
    {
      maxval = fc[i];
      maxid = i;
    }
  }
  return maxid;
}

void vggnet(float * images, float * network, int * labels, float * confidences, int num_images)
{
  int ndev = cl_setting(network);

  float *p5[16];
  float *fc1[16], *fc2[16], *fc3[16]; // Fully connected layers
  int i, j;

  for (i = 0; i < 16; i ++) {
    p5[i] = (float *)malloc(sizeof(float) * 7 * 7 * 512);

    fc1[i] = (float *)malloc(sizeof(float) * 4096);
    fc2[i] = (float *)malloc(sizeof(float) * 4096);
    fc3[i] = (float *)malloc(sizeof(float) * 1000);
  }

  cl_int error;
  for(i = 0; i < num_images; i += ndev)
  {
    int device_num = 16;
	float *image[16];
	for (j = 0; j < ndev; j++) {
	  if (i + j < num_images) {        
	    image[j] = images + (i + j) * 224 * 224 * 3;
		error = clEnqueueWriteBuffer(command_queue[j], inout1_buf[j], CL_FALSE, 0, 224 * 224 * 3 * sizeof(float), (void *) ((size_t) image[j]), 0, NULL, NULL);
        enqueue_buffer_error_check(error, "vggnet-convolution", "first_input");
	  } else {
	    device_num = j;
		break;
	  }
	}

    convolution_layer_buf(inout1_buf, inout2_buf, mats_buf, f1_1_buf, b1_1_buf, 224, 3, 64, device_num);
//test
/*
	if (i == 0) {
	  float *c1_1_t = (float *)malloc(sizeof(float) * 224 * 224 * 64);
      convolution_layer_d(image[0], c1_1_t, f1_1, b1_1, 224, 3, 64);
	  int dif = 0;
	  int it;
	  for (it = 0; it < 224; it++) {
	    if (c1_1[0][it] != c1_1_t[it]) {
			  dif = 1;
			  printf("%d'th element of conv is diffent. %f, correct %f\n", it, c1_1[0][it], c1_1_t[it]);
	    }
	  }
	  if (dif == 0) {
	    printf("no difference!\n");
	  }
	}
*/
    convolution_layer_buf(inout2_buf, inout1_buf, mats_buf, f1_2_buf, b1_2_buf, 224, 64, 64, device_num);
    pooling_layer_buf(inout1_buf, inout2_buf, 112, 64, device_num); 
//test
/*
		if (i == 0) {
		  float *p1_t = (float *)malloc(sizeof(float) * 112 * 112 * 64);
		  pooling_layer_d(c1_2, p1_t, 112, 64);
		  int dif = 0;
		  int it;
		  for (it = 0; it < 64; it++) {
		    if (p1[it] != p1_t[it]) {
				  dif = 1;
				  printf("%d'th element of pooling is different. %f, correct %f\n", it, p1[it], p1_t[it]);
				}
		  }
		  if (dif == 0) {
		    printf("no difference!\n");
		  }
		}
*/

    convolution_layer_buf(inout2_buf, inout1_buf, mats_buf, f2_1_buf, b2_1_buf, 112, 64, 128, device_num);
    convolution_layer_buf(inout1_buf, inout2_buf, mats_buf, f2_2_buf, b2_2_buf, 112, 128, 128, device_num);
    pooling_layer_buf(inout2_buf, inout1_buf, 56, 128, device_num);

    convolution_layer_buf(inout1_buf, inout2_buf, mats_buf, f3_1_buf, b3_1_buf, 56, 128, 256, device_num);
    convolution_layer_buf(inout2_buf, inout1_buf, mats_buf, f3_2_buf, b3_2_buf, 56, 256, 256, device_num);
    convolution_layer_buf(inout1_buf, inout2_buf, mats_buf, f3_3_buf, b3_3_buf, 56, 256, 256, device_num);
    pooling_layer_buf(inout2_buf, inout1_buf, 28, 256, device_num);

    convolution_layer_buf(inout1_buf, inout2_buf, mats_buf, f4_1_buf, b4_1_buf, 28, 256, 512, device_num);
    convolution_layer_buf(inout2_buf, inout1_buf, mats_buf, f4_2_buf, b4_2_buf, 28, 512, 512, device_num);
    convolution_layer_buf(inout1_buf, inout2_buf, mats_buf, f4_3_buf, b4_3_buf, 28, 512, 512, device_num);
    pooling_layer_buf(inout2_buf, inout1_buf, 14, 512, device_num);

    convolution_layer_buf(inout1_buf, inout2_buf, mats_buf, f5_1_buf, b5_1_buf, 14, 512, 512, device_num);
    convolution_layer_buf(inout2_buf, inout1_buf, mats_buf, f5_2_buf, b5_2_buf, 14, 512, 512, device_num);
    convolution_layer_buf(inout1_buf, inout2_buf, mats_buf, f5_3_buf, b5_3_buf, 14, 512, 512, device_num);
    pooling_layer_buf(inout2_buf, inout1_buf, 7, 512, device_num);

	for (j = 0; j < device_num; j++) {
	  error = clEnqueueReadBuffer(command_queue[j], inout1_buf[j], CL_TRUE, 0, 7 * 7 * 512 * sizeof(float), (void *) ((size_t) p5[j]), 0, NULL, NULL);
      enqueue_buffer_error_check(error, "vggnet-convolution", "p5");

	  fc_layer_d(p5[j], fc1[j], w1, b1, 7 * 7 * 512, 4096); 

//test
/*
    if (i == 0) {
		  float *fc1_t = (float *)malloc(sizeof(float) * 4096);
      fc_layer_d(p5, fc1_t, w1, b1, 7 * 7 * 512, 4096);
      int dif = 0;
      int it;
      for (it = 0; it < 64; it++) {
        if (fc1[it] != fc1_t[it]) {
          dif = 1;
          printf("%d'th element of fc is different. %f, correct %f\n", it, fc1[it], fc1_t[it]);
        }
      }
      if (dif == 0) {
        printf("no difference!\n");
      }
    }
*/

      fc_layer_d(fc1[j], fc2[j], w2, b2, 4096, 4096);
      fc_layer_d(fc2[j], fc3[j], w3, b3, 4096, 1000);

	  softmax(fc3[j]);
    }

    for (j = 0; j < ndev; j++) {
      if (i + j < num_images) {
        labels[i + j] = find_max(fc3[j]);
        confidences[i + j] = fc3[j][labels[i + j]];
      } else {
        break;
      }
    }
  }

  for (i = 0; i < 16; i++) {
    free(p5[i]);

    free(fc1[i]);
    free(fc2[i]);
    free(fc3[i]);
  }
}

void build_program_check(cl_int error, cl_program program, cl_device_id *device) {
  if (error != CL_SUCCESS) {
    printf("Program build fail\n");
    size_t len;
    char buffer[2048];
    clGetProgramBuildInfo(program, *device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
    printf("%s\n", buffer);
    exit(1);
  }
}

void create_kernel_error_check(cl_int error, char *where) {
  if (error != CL_SUCCESS) {
    printf("kernel creation in %s layer faile\n", where);
    switch (error) {
      case CL_INVALID_PROGRAM:
        printf("Invalid rogram\n");
        break;
      case CL_INVALID_PROGRAM_EXECUTABLE:
        printf("Invalid program executable\n");
        break;
      case CL_INVALID_KERNEL_NAME:
        printf("kerenel name is not found in program\n");
        break;
      default:
        printf("Not in list\n");
    }
    exit(1);
  }
}

void enqueue_kernel_error_check(cl_int error, char *where) {
  if (error != CL_SUCCESS) {
    printf("NDRange kernel enqueue in %s failed\n", where);
    switch (error) {
      case CL_INVALID_PROGRAM_EXECUTABLE:
        printf("Invalid program executable\n");
        break;
      case CL_INVALID_COMMAND_QUEUE:
        printf("Invalid command queue\n");
        break;
      case CL_INVALID_KERNEL:
        printf("Invalid kernel\n");
        break;
      case CL_INVALID_CONTEXT:
        printf("Invalid context\n");
        break;
      case CL_INVALID_KERNEL_ARGS:
        printf("Invalid kernel args\n");
        break;
      case CL_INVALID_WORK_DIMENSION:
        printf("Invalid work dimension\n");
        break;
      case CL_INVALID_WORK_GROUP_SIZE:
        printf("Invalid work group size\n");
        break;
      case CL_INVALID_WORK_ITEM_SIZE:
        printf("Invalid work item size\n");
        break;
      case CL_INVALID_GLOBAL_OFFSET:
        printf("Invalid global offset\n");
        break;
      case CL_OUT_OF_RESOURCES:
        printf("Out of resources\n");
        break;
      case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        printf("Mem object allocation fail\n");
        break;
      case CL_INVALID_EVENT_WAIT_LIST:
        printf("Invalid event wait list\n");
        break;
      case CL_OUT_OF_HOST_MEMORY:
        printf("Out of host mem\n");
        break;
      default:
        printf("Not in list\n");
    }
    exit(1);
  }
}

void enqueue_buffer_error_check(cl_int error, char *where, char *what) {
  if (error != CL_SUCCESS) {
    printf("%s buffer enqueue failed in %s\n", what, where);
    switch (error) {
      case CL_INVALID_COMMAND_QUEUE:
        printf("Invalid command q\n");
        break;
      case CL_INVALID_CONTEXT:
        printf("Invalid context\n");
        break;
      case CL_INVALID_MEM_OBJECT:
        printf("Invalid memory obj\n");
        break;
      case CL_INVALID_VALUE:
        printf("Invalid value\n");
        break;
      case CL_INVALID_EVENT_WAIT_LIST:
        printf("invalid event wait list\n");
        break;
      case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        printf("Mem object alloc fail\n");
        break;
      case CL_OUT_OF_HOST_MEMORY:
        printf("Out of host mem\n");
        break;
      default:
        printf("Not in list\n");
    }
    exit(1);
  }
}

void conv_single_itr_test(float *input_tmp, float *output_tmp, float *filter_tmp, int N) {
  float in1[4] = { input_tmp[0], input_tmp[1], input_tmp[N], input_tmp[N + 1] };
  float in2[9] = { input_tmp[2 * N + 2], input_tmp[2 * N + 3], input_tmp[2 * N + 4], input_tmp[3 * N + 2], input_tmp[3 * N + 3], input_tmp[3 * N + 4], input_tmp[4 * N + 2], input_tmp[4 * N + 3], input_tmp[4 * N + 4] };
  float in3[4] = { input_tmp[(N - 2) * N + N - 2], input_tmp[(N - 2) * N + N - 1], input_tmp[(N - 1) * N + N - 2], input_tmp[(N - 1) * N + N - 1] };
  float fi1[4] = { filter_tmp[4], filter_tmp[5], filter_tmp[7], filter_tmp[8] };
  float fi2[9] = { filter_tmp[0], filter_tmp[1], filter_tmp[2], filter_tmp[3], filter_tmp[4], filter_tmp[5], filter_tmp[6], filter_tmp[7], filter_tmp[8] };
  float fi3[4] = { filter_tmp[0], filter_tmp[1], filter_tmp[3], filter_tmp[4] };
  float mul1 = 0;
  int k;
  for (k = 0; k < 4; k++)
    mul1 += in1[k] * fi1[k];
  if (output_tmp[0] != mul1) {
    printf("4 input val for 0 0: %f, %f, %f, %f\n", in1[0], in1[1], in1[2], in1[3]);
    printf("4 filter val for 0 0: %f, %f, %f, %f\n", fi1[0], fi1[1], fi1[2], fi1[3]);
    printf("Convolution incorrect result for 0 0: %f, mul result: %f\n", output_tmp[0], mul1);
  } else {
    printf("Convolution correct result for 0 0\n");
  }

  float mul2 = 0;
  for (k = 0; k < 9; k++)
    mul2 += in2[k] * fi2[k];
  if (output_tmp[3 * N + 3] != mul2) {
    printf("9 input val for 3 3: %f, %f, %f, %f, %f, %f, %f, %f, %f\n", in2[0], in2[1], in2[2], in2[3], in2[4], in2[5], in2[6], in2[7], in2[8]);
    printf("9 filter val for 3 3: %f, %f, %f, %f, %f, %f, %f, %f, %f\n", fi2[0], fi2[1], fi2[2], fi2[3], fi2[4], fi2[5], fi2[6], fi2[7], fi2[8]);
    printf("Convolution incorrect result for 3 3: %f, mul result: %f\n", output_tmp[3 * N + 3], mul2);
  } else {
    printf("Convolution correct result for 3 3\n");
  }

  float mul3 = 0;
  for (k = 0; k < 4; k++)
    mul3 += in3[k] * fi3[k];
  if (output_tmp[(N - 1) * N + (N - 1)] != mul3) {
    printf("4 input val for 0 0: %f, %f, %f, %f\n", in3[0], in3[1], in3[2], in3[3]);
    printf("4 filter val for 0 0: %f, %f, %f, %f\n", fi3[0], fi3[1], fi3[2], fi3[3]);
    printf("Convolution result for %d, %d: %f, mul result: %f\n\n", N - 1, N - 1, output_tmp[(N - 1) * N + (N - 1)], mul3);
  } else {
    printf("Convolution correct result for %d %d\n\n", N-1, N-1);
  }
}
