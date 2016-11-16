TARGET=mat_mul
OBJS=mat_mul.o timers.o

CC=gcc
CFLAGS=-g -O3 -Wall
LDFLAGS= -lOpenCL

all: $(TARGET)

$(TARGET):$(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o $@

clean:
	rm -rf $(TARGET) $(OBJS) task*

run: $(TARGET)
	thorq --add --device gpu --timeout 720 ./$(TARGET) 8192 4 1 -v
