INC_DIR = -I /home/dcs/csswdm/GraphBLAS/Include/
LIB_DIR = -L /home/dcs/csswdm/GraphBLAS/build/
LIBS = -lgraphblas -lm -lrt
CFLAGS = -O3 -Wall
CC = mpicc
SRCS=$(wildcard *.c)
OBJS=$(SRCS:.c=.o)

cagnet1d: $(OBJS)
	$(CC) $(CFLAGS) $(INC_DIR) $(LIB_DIR) $? -o $@ $(LIBS) 
.c.o:
	$(CC) $(CFLAGS) $(INC_DIR) -c $<
clean:
	-rm -f *.o
	-rm -f cagnet1d
