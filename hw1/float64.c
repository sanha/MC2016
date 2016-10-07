#include <stdio.h>
#include <stdlib.h>

int main() {
  char buf[65];
  long *input;
  input = malloc(sizeof(long));
  long tmp;

  scanf("%lf", input);
  tmp = *input;
  
  int i;
  char k;
  for (i=0; i<64; i++) {
    k = tmp & 1 ? '1' : '0';
    buf[63 - i] = k;
    tmp >>= 1;
  }
  buf[65] = '\0';

  puts(buf);
  double *f;
  f = input;

  printf("%f\n", *f);

  free(input);

  return 0;
}
