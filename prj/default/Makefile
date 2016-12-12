
CC=gcc
CFLAGS=-Wall -g -O3

LIBS = -lm -lrt -lOpenCL
LDFLAGS = ${LIBS}

all: vggnet

.PHONY: all vggnet

vggnet: vggnet.o main.o
	${CC} $^ -o $@ ${LDFLAGS}
run:
	thorq --add --device cpu ./vggnet image_list.txt

clean:
	rm -f vggnet.o main.o vggnet task*
