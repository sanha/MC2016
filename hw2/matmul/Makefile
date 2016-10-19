TARGET=mat_mul
OBJS=mat_mul.o timers.o

CC=gcc
CFLAGS=-g -O3 -Wall
LDFLAGS= -lpthread

all: $(TARGET)

$(TARGET):$(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o $@

clean:
	rm -rf $(TARGET) $(OBJS)

clog:
	rm -rf task*

run: $(TARGET)
	thorq --add --timeout 120 ./$(TARGET) -v
