
CC=gcc
CFLAGS=-Wall -g -O3

LIBS = -lm -lrt -lOpenCL
LDFLAGS = ${LIBS}

all: vggnet

.PHONY: all vggnet

vggnet: vggnet.o main.o
	${CC} $^ -o $@ ${LDFLAGS}
run:
	thorq --add --device cpu --timeout 3600 ./vggnet image_list.txt

profile:
	thorq --add --device cpu --timeout 3600 /opt/AMDAPP/AMDAPPProfiler-2.5/x86_64/sprofile -p -w . ./vggnet image_list.txt

clean:
	rm -f vggnet.o main.o vggnet task*
