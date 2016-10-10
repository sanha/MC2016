#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>
#include <assert.h>

int main(void) {

  double x, y, result1, result2;
  float fma_time_accum = 0;
  float non_fma_time_accum = 0;
  struct timeval time1, time2, time3;
  srand(time(NULL));
  x = (float) rand() / (float) RAND_MAX;
  y = (float) rand() / (float) RAND_MAX;

  int i, j;
  int outer_itr = 20;
  int iterate = 1e8;
  for(i = 0; i < outer_itr; i++) {
    gettimeofday(&time1, NULL);
    for(j = 0; j < iterate; j++) {
      result1 = fma(x, (float)j, y);
      result1 = fma(result1, x, y);
      result1 = fma(result1, y, x);
    }
    gettimeofday(&time2, NULL);
    for(j = 0; j < iterate; j++) {
      result2 = x * (float)j + y;
      result2 = result2 * x + y;
      result2 = result2 * y + x;
    }
    gettimeofday(&time3, NULL);
    
    assert(result1 == result2);

    float fma_time = time2.tv_sec + time2.tv_usec * 1e-6 - time1.tv_sec - time1.tv_usec * 1e-6;
    float non_fma_time = time3.tv_sec + time3.tv_usec * 1e-6 - time2.tv_sec - time2.tv_usec * 1e-6;

    fma_time_accum += fma_time;
    non_fma_time_accum += non_fma_time;
    printf("fma time consuming in iterate %d: %f\n", i + 1, fma_time);
    printf("non-fma time consuming in iterate %d: %f\n", i + 1, non_fma_time);
  }
  printf("total fma time consuming: %f\n", fma_time_accum);
  printf("total non-fma time consuming: %f\n", non_fma_time_accum);
  printf("%.2f percent of total computation time is saved.\n", (1 - fma_time_accum / non_fma_time_accum) * 100);
}
