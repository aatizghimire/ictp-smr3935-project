CC = gcc
CFLAGS = -Wall -Wextra -std=c99
LDFLAGS =

SRC = main.c neural_network.c
OBJ = $(SRC:.c=.o)
EXECUTABLE = neural_network

.PHONY: all clean

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(EXECUTABLE)
