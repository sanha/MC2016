#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

const char* pooling_kernel_src = 
				"__kernel void pooling(__global float *input, " \
				"__global float *output, " \
				"int N) " \
				"{" \
				"	int g_id1 = get_global_id(0);" \
				"	int g_id2 = get_global_id(1);" \
				"	int i, j;" \
				"	float max = 0;" \
				"	int offset;" \
				"	for (i = 0; i < 2; i++) { " \
				"		offset = (g_id1 * 2 + i) * 2 * N + g_id2 * 2;" \
				"		for (j = 0; j < 2; j++) { " \
				"			float pixel = input[offset + j]; " \
				"			max = (max > pixel) ? max : pixel; " \
				"		}" \
				"	}" \
				"	output[g_id1 * N + g_id2] = max;" \
				"}";

static void pooling_layer(float * inputs, float * outputs, int N, int D, cl_device_id *device, cl_context context, cl_command_queue *command_queue, int ndev, cl_program program)
{
  cl_int error;	
  int input_size = 4 * N * N * sizeof(float) / ndev;
  int output_size = N * N * sizeof(float) / ndev;
  int i, j;
  int n = N;
  size_t global[2] = {N, N};
  size_t local[2];
  int work_group_size;

  switch (N) {
    case 112:
	  work_group_size = 16;
	  break;
	case 56:
	  work_group_size = 8;
	  break;
	default:
	  work_group_size = 7;
  }

  local[0] = work_group_size;
  local[1] = work_group_size;
  
  for (i = 0; i < D; i ++) {
    cl_mem input_buf[ndev];
    cl_mem output_buf[ndev];

    for (j = 0; j < ndev; j++) {
      input_buf[j] = clCreateBuffer(context, CL_MEM_READ_ONLY, input_size, NULL, NULL);	
      output_buf[j] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_size, NULL, NULL);
    }

    cl_kernel kernel[ndev];
    for (j = 0; j < ndev; j++) {
  	  kernel[j] = clCreateKernel(program, "pooling", &error);
	  if (error != CL_SUCCESS) {
	    printf("i %d, j %d kernel creation in pooling layer faile\n", i, j);
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

    for (j = 0; j < ndev; j++) {
      clSetKernelArg(kernel[j], 0, sizeof(cl_mem), (void *)&input_buf[j]);
  	  clSetKernelArg(kernel[j], 1, sizeof(cl_mem), (void *)&output_buf[j]);
  	  clSetKernelArg(kernel[j], 2, sizeof(int), (void *)&n);
    }

    for (j = 0; j < ndev; j++) {
      error = clEnqueueWriteBuffer(command_queue[j], input_buf[j], CL_FALSE, 0, input_size, (void *) ((size_t) inputs + i * N * N * 4 *sizeof(float) + j * input_size), 0, NULL, NULL);
      if (error != CL_SUCCESS) {
	    printf("%d'th input buffer enqueue in pooling layer failed\n", j);
  	    exit(1);
	  }
    }

	for (j = 0; j < ndev; j++) {
	  error = clEnqueueNDRangeKernel(command_queue[j], kernel[j], 2, NULL, global, local, 0, NULL, NULL);
	  if (error != CL_SUCCESS) {
	    printf("i %d, j %d NDRange kernel enqueue in pooling failed\n", i, j);
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

	for (j = 0; j < ndev; j++) {
	  error = clEnqueueReadBuffer(command_queue[j], output_buf[j], CL_TRUE, 0, output_size, (void *) ((size_t) outputs + i * N * N * sizeof(float) + j * output_size), 0, NULL, NULL);
	  if (error != CL_SUCCESS) {
	    printf("%d'th output buffer enqueue in pooling failed\n", j);
	    exit(1);
	  }
	}
  }

/*
  for(i = 0; i < D; i++)
  {
    float * input = inputs + i * N * N * 4;
    float * output = outputs + i * N * N;
    pooling2x2(input, output, N);
  }*/
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
        for(l = 0; l < 3; l++)
        {
          int x = i + k - 1;
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
static void convolution_layer(float * inputs, float * outputs, float * filters, float * biases, int N, int D1, int D2)
{
  int i, j;

  memset(outputs, 0, sizeof(float) * N * N * D2);

  for(j = 0; j < D2; j++)
  {
    for(i = 0; i < D1; i++)
    {
      float * input = inputs + N * N * i;
      float * output = outputs + N * N * j;
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

static void fc_layer(float * input_neuron, float * output_neuron, float * weights, float * biases, int N, int M)
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


static float * get_param(float ** array, int size)
{
  float * subarray = *array;
  *array += size;
  return subarray;
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

void vggnet(float * images, float * network, int * labels, float * confidences, int num_images)
{
  float *c1_1, *c1_2, *c2_1, *c2_2, *c3_1, *c3_2, *c3_3, *c4_1, *c4_2, *c4_3, *c5_1, *c5_2, *c5_3; // Convolution layers
  float *p1, *p2, *p3, *p4, *p5; // Pooling layers
  float *fc1, *fc2, *fc3; // Fully connected layers
  float *f1_1, *f1_2, *f2_1, *f2_2, *f3_1, *f3_2, *f3_3, *f4_1, *f4_2, *f4_3, *f5_1, *f5_2, *f5_3, *w1, *w2, *w3; // Filters and weights
  float *b1_1, *b1_2, *b2_1, *b2_2, *b3_1, *b3_2, *b3_3, *b4_1, *b4_2, *b4_3, *b5_1, *b5_2, *b5_3, *b1, *b2, *b3; // Biases
  int i;

  c1_1 = (float *)malloc(sizeof(float) * 224 * 224 * 64);
  c1_2 = (float *)malloc(sizeof(float) * 224 * 224 * 64);

  p1 = (float *)malloc(sizeof(float) * 112 * 112 * 64);

  c2_1 = (float *)malloc(sizeof(float) * 112 * 112 * 128);
  c2_2 = (float *)malloc(sizeof(float) * 112 * 112 * 128);

  p2 = (float *)malloc(sizeof(float) * 56 * 56 * 128);

  c3_1 = (float *)malloc(sizeof(float) * 56 * 56 * 256);
  c3_2 = (float *)malloc(sizeof(float) * 56 * 56 * 256);
  c3_3 = (float *)malloc(sizeof(float) * 56 * 56 * 256);

  p3 = (float *)malloc(sizeof(float) * 28 * 28 * 256);

  c4_1 = (float *)malloc(sizeof(float) * 28 * 28 * 512);
  c4_2 = (float *)malloc(sizeof(float) * 28 * 28 * 512);
  c4_3 = (float *)malloc(sizeof(float) * 28 * 28 * 512);

  p4 = (float *)malloc(sizeof(float) * 14 * 14 * 512);

  c5_1 = (float *)malloc(sizeof(float) * 14 * 14 * 512);
  c5_2 = (float *)malloc(sizeof(float) * 14 * 14 * 512);
  c5_3 = (float *)malloc(sizeof(float) * 14 * 14 * 512);

  p5 = (float *)malloc(sizeof(float) * 7 * 7 * 512);

  fc1 = (float *)malloc(sizeof(float) * 4096);
  fc2 = (float *)malloc(sizeof(float) * 4096);
  fc3 = (float *)malloc(sizeof(float) * 1000);

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

  cl_int error;
  int ndev;

  cl_platform_id platform;
  error = clGetPlatformIDs(1, &platform, NULL);
  if (error != CL_SUCCESS) {
  	printf("Get platform id fail\n");
	exit(1);
  }

  error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, (unsigned int *) &ndev);
  if (error != CL_SUCCESS) {
  	printf("Get cpu number fail\n");
	exit(1);
  }
  printf("cpu device num: %d\n", ndev);

  cl_device_id device[ndev];
  error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, ndev, device, NULL);
  if (error != CL_SUCCESS) {
  	printf("Get cpu id fail\n");
	exit (1);
  }

  cl_context context;
  context = clCreateContext(0, ndev, device, NULL, NULL, &error);
  if (error != CL_SUCCESS) {
  	printf("Context creation failed\n");
	exit(1);
  }

  cl_command_queue command_queue[ndev];
  for (i = 0; i < ndev; i++) {
  	command_queue[i] = clCreateCommandQueue(context, device[i], 0, &error);
	if (error != CL_SUCCESS) {
	  printf("%d'th command queue creation failed\n", i);
	}
  }

  cl_program pooling_program;
  size_t pooling_kernel_src_len = strlen(pooling_kernel_src);
  pooling_program = clCreateProgramWithSource(context, 1, (const char **) &pooling_kernel_src, &pooling_kernel_src_len, NULL);

  error = clBuildProgram(pooling_program, ndev, device, NULL, NULL, NULL);
  build_program_check(error, pooling_program, device);

  for(i = 0; i < num_images; i++)
  {
    float * image = images + i * 224 * 224 * 3;

    convolution_layer(image, c1_1, f1_1, b1_1, 224, 3, 64);
    convolution_layer(c1_1, c1_2, f1_2, b1_2, 224, 64, 64);
    pooling_layer(c1_2, p1, 112, 64, device, context, command_queue, ndev, pooling_program);

    convolution_layer(p1, c2_1, f2_1, b2_1, 112, 64, 128);
    convolution_layer(c2_1, c2_2, f2_2, b2_2, 112, 128, 128);
    pooling_layer(c2_2, p2, 56, 128, device, context, command_queue, ndev, pooling_program);

    convolution_layer(p2, c3_1, f3_1, b3_1, 56, 128, 256);
    convolution_layer(c3_1, c3_2, f3_2, b3_2, 56, 256, 256);
    convolution_layer(c3_2, c3_3, f3_3, b3_3, 56, 256, 256);
    pooling_layer(c3_3, p3, 28, 256, device, context, command_queue, ndev, pooling_program);

    convolution_layer(p3, c4_1, f4_1, b4_1, 28, 256, 512);
    convolution_layer(c4_1, c4_2, f4_2, b4_2, 28, 512, 512);
    convolution_layer(c4_2, c4_3, f4_3, b4_3, 28, 512, 512);
    pooling_layer(c4_3, p4, 14, 512, device, context, command_queue, ndev, pooling_program);

    convolution_layer(p4, c5_1, f5_1, b5_1, 14, 512, 512);
    convolution_layer(c5_1, c5_2, f5_2, b5_2, 14, 512, 512);
    convolution_layer(c5_2, c5_3, f5_3, b5_3, 14, 512, 512);
    pooling_layer(c5_3, p5, 7, 512, device, context, command_queue, ndev, pooling_program);

    fc_layer(p5, fc1, w1, b1, 7 * 7 * 512, 4096);
    fc_layer(fc1, fc2, w2, b2, 4096, 4096);
    fc_layer(fc2, fc3, w3, b3, 4096, 1000);

    softmax(fc3);

    labels[i] = find_max(fc3);
    confidences[i] = fc3[labels[i]];
  }

  free(c1_1);
  free(c1_2);
  free(p1);

  free(c2_1);
  free(c2_2);
  free(p2);

  free(c3_1);
  free(c3_2);
  free(c3_3);
  free(p3);

  free(c4_1);
  free(c4_2);
  free(c4_3);
  free(p4);

  free(c5_1);
  free(c5_2);
  free(c5_3);
  free(p5);

  free(fc1);
  free(fc2);
  free(fc3);
}
