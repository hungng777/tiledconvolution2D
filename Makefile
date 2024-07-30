# IDIR=./
CXX = nvcc

CXXFLAGS += $(shell pkg-config --cflags --libs opencv4)
LDFLAGS += $(shell pkg-config --libs --static opencv)

all: clean build

build: 
	$(CXX) tiledConvolution2D.cu --std c++17 `pkg-config opencv --cflags --libs` -o tiledConvolution2D.exe -Wno-deprecated-gpu-targets $(CXXFLAGS) -I/usr/local/cuda/include -lcuda

run:
	./tiledConvolution2D.exe $(ARGS)

clean:
	rm -f tiledConvolution2D.exe