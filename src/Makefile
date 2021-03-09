# This Makefile will attempt to compile any file
# in the current directory with the suffix `.cpp`.

CXX = upcxx

SOURCES += $(wildcard *.cpp)
TARGETS := $(patsubst %.cpp, %, $(SOURCES))

CXXFLAGS = -std=gnu++17 -O3

all: $(TARGETS)

%: %.cpp
	$(CXX) -o $@ $^ $(CXXFLAGS)

clean:
	rm -fv $(TARGETS)

