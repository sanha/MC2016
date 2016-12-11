TARGET1=mat_mul
TARGET2=int_mat_mul
COMMON=timers.o
OBJS1=mat_mul.o
OBJS2=int_mat_mul.o

CC=gcc
CFLAGS=-g -O2 -Wall
AUTOFLAGS=-c -O2 -Wall -ftree-vectorize

all: $(OBJS1) $(TARGET1) $(TARGET2)

$(OBJS1):
	$(CC) $(AUTOFLAGS) -o $@ mat_mul.c

$(TARGET1):$(OBJS1) $(COMMON)
	$(CC) $(OBJS1) $(COMMON) -o $@

$(TARGET2):$(OBJS2) $(COMMON)
	$(CC) $(OBJS2) $(COMMON) -o $@

clean:
	rm -rf $(TARGET1) $(TARGET2) $(OBJS1) $(OBJS2) $(COMMON) task*

run: $(TARGET1)
	thorq --add --device gpu --timeout 720 ./$(TARGET1) 2048 -v

runint: $(TARGET2)
	thorq --add --device gpu --timeout 720 ./$(TARGET2) 2048 -v
