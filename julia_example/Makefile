# Compile the my CUDA examples

CC=/opt/gcc-4.6.4/bin/gcc
CXX=/opt/gcc-4.6.4/bin/g++
CUDACC=nvcc

CFLAGS=-ggdb
CUDACFLAGS=-m64

LDFLAGS=-L/usr/local/cuda/lib64 -lcuda -lcublas -lcudart 
RM=rm -rf

CUDA_SOURCES=julia.cu
SOURCES=julia_main.cpp
CUDA_OBJECTS=$(CUDA_SOURCES:.cu=.o)
OBJECTS=$(SOURCES:.cpp=.o)

.SUFFIXES: .cpp .cu .o

all: julia

julia: julia.h $(OBJECTS) $(CUDA_OBJECTS)
	$(CXX) -lopencv_core -lopencv_highgui -lopencv_imgproc $(LDFLAGS) $(CUDA_OBJECTS) $(OBJECTS) -o julia

clean:
	$(RM) *.o julia

.cpp.o:
	$(CXX) $(CFLAGS) $< -c -o $@

.cu.o:
	$(CUDACC) --compiler-bindir /opt/gcc-4.6.4/bin $(CUDACFLAGS) $< -c -o $@
