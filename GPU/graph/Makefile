
INC_DIR = -I ./lib/include/
LIB_DIR = -L ./lib/lib/
LIBS = -lmetis
CFLAGS = -g -Wall -O3 -std=c++11
CC = g++
SRCS=$(wildcard *.cpp)
OBJS=$(SRCS:.cpp=.o)

gcngp: $(OBJS)
	$(CC) -fopenmp $(CFLAGS) $(INC_DIR) $(LIB_DIR) $? -o $@ $(LIBS) 
.cpp.o:
	$(CC) -fopenmp $(CFLAGS) $(INC_DIR) -c $<
clean:
	-rm -f *.o
	-rm -f gcngp

