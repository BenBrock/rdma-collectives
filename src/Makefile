
# This Makefile will attempt to compile any file
# in the current directory with the suffix `.cpp`.

CXX = upcxx

SOURCES += $(wildcard *.cpp)
TARGETS := $(patsubst %.cpp, %, $(SOURCES))


all: $(TARGETS)

%: %.cpp
	$(CXX) -O -o $@ $^

clean:
	rm -fv $(TARGETS)