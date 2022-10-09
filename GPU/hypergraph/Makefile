
INC_DIR = -I ./lib/Linux-x86_64/
LIB_DIR = -L ./lib/Linux-x86_64/
LIBS = -lpatoh
CFLAGS = -g -Wall -O3 -std=c++11
CC = g++
SRCS=$(wildcard *.cpp)
OBJS=$(SRCS:.cpp=.o)

gcnhgp: $(OBJS)
	$(CC) -fopenmp $(CFLAGS) $(INC_DIR) $(LIB_DIR) $? -o $@ $(LIBS) 
.cpp.o:
	$(CC) -fopenmp $(CFLAGS) $(INC_DIR) -c $<
clean:
	-rm -f *.o
	-rm -f gcnhgp

