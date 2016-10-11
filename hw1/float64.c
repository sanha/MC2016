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
  char buf[65];
  long *input = malloc(sizeof(long));
  unsigned long tmp;

  // get input as a float
  scanf("%lf", input);
  tmp = *input;
  
  // convert it to binary
  int i;
  char k;
  for (i = 0; i < 64; i++) {
    buf[63 - i] = tmp & 1 ? '1' : '0';
    tmp >>= 1;
  }
  buf[64] = '\0';
  puts(buf);

  // convert binary to floating point expression
  int sign = buf[0] == '1' ? -1 : 1;
  int bias = 1023;
  unsigned short *e = malloc(sizeof(unsigned short));
  memcpy(e, (void *)input + 6, 2);
  *e <<= 1;
  *e >>= 5;
  unsigned long *f = calloc(1, sizeof(unsigned long));
  memcpy(f, input, 7);
  *f <<= 12;
  *f >>= 12;
  long double d = *f;
  for (i = 0; i < 52; i++) {
    d /= 2.0;
  }

  // print result
  double result;
  if (*e == 0) {
    if (*f == 0) {
      // represents zero
      result = sign == 1 ? 0.0 : -0.0;
    }
    else {
      // represents subnormal value
      result = sign * d * power_of_two(1 - bias);
    }
  } else if (*e == 2047) {
    if (*f == 0) {
      // represents infinite
      result = sign == 1 ? 1.0 / 0.0 : -1.0 / 0.0;
    } else {
      // represents 
      result = sign == 1? NAN : -NAN;
    }
  } else {
    result = sign * (d + 1) * power_of_two(*e - bias);
  }
  printf("%lf\n", result);
  
  // Test code: to compare
  /* 
  double *fpr;
  fpr = input;
  printf("%lf (for test)\n", *fpr);
  */

  free(input);
  free(e);
  free(f);

  return 0;
}
