#include <stdio.h>
#include <stdlib.h>

int main() {
  char buf[33];
  int *input;
  input = malloc(sizeof(int));
  int tmp;

  scanf("%f", input);
  tmp = *input;
  
  int i;
  char k;
  for (i=0; i<32; i++) {
    k = tmp & 1 ? '1' : '0';
    buf[31 - i] = k;
    tmp >>= 1;
  }
  buf[32] = '\0';

  puts(buf);
  float *f;
  f = input;

  printf("%f\n", *f);

  free(input);

  return 0;
}
