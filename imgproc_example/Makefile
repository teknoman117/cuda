# Compile the my CUDA examples

CUDACC=nvcc
CUDACFLAGS=

RM=rm -rf

SOURCES=main.cu
OBJECTS=$(SOURCES:.cu=.o)
LDFLAGS=-lopencv_core -lopencv_highgui -lopencv_imgproc

.SUFFIXES: .cu .o

all: imgproc

imgproc: $(OBJECTS)
	$(CUDACC) $(LDFLAGS) $(OBJECTS) -o imgproc

clean:
	$(RM) *.o imgproc

.cu.o:
	$(CUDACC) $(CUDACFLAGS) $< -c -o $@
