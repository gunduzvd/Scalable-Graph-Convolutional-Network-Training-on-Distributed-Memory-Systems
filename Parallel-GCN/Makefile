INC_DIR = -I ../GraphBLAS/Include/
LIB_DIR = -L ../GraphBLAS/build/
LIBS = -lgraphblas -lm -lrt
CFLAGS = -O3 -Wall
CC = mpicc
SRCS=$(wildcard *.c)
OBJS=$(SRCS:.c=.o)

grbgcn: $(OBJS)
	$(CC) $(CFLAGS) $(INC_DIR) $(LIB_DIR) $? -o $@ $(LIBS) 
.c.o:
	$(CC) $(CFLAGS) $(INC_DIR) -c $<
clean:
	-rm -f *.o
	-rm -f grbgcn
