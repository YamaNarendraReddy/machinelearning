CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -fopenmp

SRCS = main.cpp Matrix.cpp Activation.cpp
OBJS = $(SRCS:.cpp=.o)
TARGET = mlp_demo

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

run: all
	./$(TARGET)

.PHONY: all clean run
