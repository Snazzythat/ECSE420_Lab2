# CMAKE FILE
CC = mpicc
CFLAGS =
LIBS =
TARGETS = grid4_4 grid512_512

default: clean $(TARGETS)
all: default

.PHONY: clean

grid4_4:
	$(CC) -o grid4_4 grid4_4.c -std=c99 -lm

grid512_512:
	$(CC) -o grid512_512 grid512_512.c -std=c99 -lm

clean:
	-rm -f $(TARGETS)
	-rm -f *.o