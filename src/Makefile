# This Makefile will attempt to compile any file
# in the current directory with the suffix `.cpp`.

CXX = upcxx

SOURCES += $(wildcard *.cpp)
TARGETS := $(patsubst %.cpp, %, $(SOURCES))

all: $(TARGETS)

%: %.cpp
	$(CXX) -O3 -o $@ -g $^

clean:
	rm -fv $(TARGETS)

