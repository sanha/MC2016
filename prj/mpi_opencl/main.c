#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "network.h"
#include "class_name.h"
#include "mpi.h"

void vggnet(float * images, float * network, int * labels, float * confidences, int num_images, int rank, int assigned);

int timespec_subtract(struct timespec*, struct timespec*, struct timespec*);

int main(int argc, char** argv) {
  float *images, *network, *confidences;
  int *labels;
  int num_images, i;
  FILE *io_file;
  int sizes[32];
  char image_files[1024][1000];
  struct timespec start, end, spent;

  int rank, size;
  int fakeArgc = 1;
  MPI_Init(&fakeArgc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  if (size != 4) {
	printf("mpi size %d instead of 4!\n", size);
	exit(1);
  }

  if (argc != 2) {
    fprintf(stderr, "Usage: %s <image list>\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  io_file = fopen(argv[1], "r");
  fscanf(io_file, "%d\n", &num_images);

  for(i = 0; i < num_images; i++)
  {
    fscanf(io_file, "%s", image_files[i]); 
  }
  fclose(io_file);

  int vggnet_size = 0;
  for(i = 0; i < 32; i++)
  {
    char filename[100];
    memset(filename, 0, 100);
    strcat(filename, "network/");
    strcat(filename, file_list[i]);
    io_file = fopen(filename, "rb");
    fseek(io_file, 0, SEEK_END); 
    sizes[i] = ftell(io_file);
    vggnet_size += sizes[i];
    fclose(io_file);
  }

  // the number of assiged images
  int assigned = ceil((double) num_images / (double) size);
  int *part_labels = (int *)malloc(sizeof(int) * assigned);
  float *part_confidences = (float *)malloc(sizeof(float) * assigned);

  images = (float *)malloc(sizeof(float) * 224 * 224 * 3 * num_images);
  network = (float *)malloc(sizeof(float) * vggnet_size); 
  // changed for allgather
  labels = (int *)malloc(sizeof(int) * assigned * size);
  confidences = (float *)malloc(sizeof(float) * assigned * size);

  int vggnet_idx = 0;
  for(i = 0; i < 32; i++)
  {
    char filename[100];
    memset(filename, 0, 100);
    strcat(filename, "network/");
    strcat(filename, file_list[i]);
    io_file = fopen(filename, "rb");
    fread(network + vggnet_idx, 1, sizes[i], io_file);
    fclose(io_file);
    vggnet_idx += sizes[i]/sizeof(float);
  }

  for(i = 0; i < num_images; i++)
  {
    io_file = fopen(image_files[i], "rb");
    if(!io_file)
    {
      printf("%s does not exist!\n", image_files[i]);
    }
    fread(images + (224 * 224 * 3) * i, 4,  224 * 224 * 3, io_file);
    fclose(io_file);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  clock_gettime(CLOCK_MONOTONIC, &start);
  vggnet(images, network, part_labels, part_confidences, num_images, rank, assigned);
  MPI_Allgather(part_labels, assigned, MPI_INT, labels, assigned, MPI_INT, MPI_COMM_WORLD);
  MPI_Allgather(part_confidences, assigned, MPI_FLOAT, confidences, assigned, MPI_FLOAT, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  clock_gettime(CLOCK_MONOTONIC, &end);
  timespec_subtract(&spent, &end, &start);

  if (rank == 0) { // all other process can print out these information.
	for(i = 0; i < num_images; i++)
    {
      printf("%s :%s : %.3f\n", image_files[i], class_name[labels[i]], confidences[i]);
    }
    printf("Elapsed time: %ld.%03ld sec\n", spent.tv_sec, spent.tv_nsec/1000/1000);
  }
  return 0;
}

int timespec_subtract(struct timespec* result, struct timespec *x, struct timespec *y) {
  if (x->tv_nsec < y->tv_nsec) {
    int nsec = (y->tv_nsec - x->tv_nsec) / 1000000000 + 1;
    y->tv_nsec -= 1000000000 * nsec;
    y->tv_sec += nsec;
  }
  if (x->tv_nsec - y->tv_nsec > 1000000000) {
    int nsec = (x->tv_nsec - y->tv_nsec) / 1000000000;
    y->tv_nsec += 1000000000 * nsec;
    y->tv_sec -= nsec;
  }
  result->tv_sec = x->tv_sec - y->tv_sec;
  result->tv_nsec = x->tv_nsec - y->tv_nsec;

  return x->tv_sec < y->tv_sec;
}
