#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// calculating power of two
double power_of_two(int p) {
  int i;
  double result = 1.0;
  if (p >= 0) {
    for (i = 0; i < p; i++) {
      result *= 2.0;
    }
  } else {
    for (i = 0; i < -p; i++) {
      result /= 2.0;
    }
  }
  return result;
}

int main() {
  char buf[33];
  int *input = malloc(sizeof(int));
  unsigned int tmp;

  // get input as a float
  scanf("%f", input);
  tmp = *input;
  
  // convert it to binary
  int i;
  for (i = 0; i < 32; i++) {
    buf[31 - i] = tmp & 1 ? '1' : '0';
    tmp >>= 1;
  }
  buf[32] = '\0';
  puts(buf);

  // convert binary to floating point expression
  int sign = buf[0] == '1' ? -1 : 1;
  int bias = 127;
  unsigned short *e = malloc(sizeof(unsigned short));
  memcpy(e, (void *)input + 2, 2);
  *e <<= 1;
  *e >>= 8;
  unsigned int *f = calloc(1, sizeof(unsigned int));
  memcpy(f, input, 3);
  *f <<= 9;
  *f >>= 9;
  double d = *f;
  for (i = 0; i < 23; i++) {
    d /= 2;
  }

  // print result
  float result;
  if (*e == 0) {
    if (*f == 0) {
      // represents zero
      result = sign == 1 ? 0.0 : -0.0;
    }
    else {
      // represents subnormal value
      result = sign * d * power_of_two(1 - bias);
    }
  } else if (*e == 255) {
    if (*f == 0) {
      // represents infinite
      result = sign == 1 ? 1.0 / 0.0 : -1.0 / 0.0;
    } else {
      // represents 
      result = sign == 1 ? NAN : -NAN;
    }
  } else {
    result = sign * (d + 1) * power_of_two(*e - bias);
  }
  printf("%f\n", result);
  
  // Test code: to compare
  /* 
  float *fpr;
  fpr = input;
  printf("%f (for test)\n", *fpr);
  */
  
  free(input);
  free(e);
  free(f);

  return 0;
}
