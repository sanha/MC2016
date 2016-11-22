TARGET=mat_mul
OBJS=mat_mul.o timers.o

CC=gcc
CFLAGS=-fopenmp -g -O3 -Wall
LDFLAGS=-fopenmp -Wall

all: $(TARGET)

$(TARGET):$(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o $@

clean:
	rm -rf $(TARGET) $(OBJS) task*

run: $(TARGET)
	thorq --add --device gpu --timeout 720 ./$(TARGET) 4096 32 -v
