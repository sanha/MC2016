TARGET=mat_mul
OBJS=mat_mul.o timers.o

CC=mpicc
CFLAGS=-g -O3 -Wall
LDFLAGS=

all: $(TARGET)

$(TARGET):$(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o $@

clean:
	rm -rf $(TARGET) $(OBJS) task*

run: $(TARGET)
	thorq --add --mode mpi --nodes 4 --slots 16 --timeout 720 ./$(TARGET) -v
