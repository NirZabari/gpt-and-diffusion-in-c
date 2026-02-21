CC   = gcc
BIN  = gpt
SRC  = gpt.c
DIFF_BIN = diffusion
DIFF_SRC = diffusion.c
FLOW_BIN = flow_matching
FLOW_SRC = flow_matching.c

MNIST_URL = https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
MNIST_GZ  = train-images-idx3-ubyte.gz
MNIST     = train-images-idx3-ubyte
LIBS = -lm

CFLAGS = -O3 -march=native -ffast-math -Wall

.PHONY: all build clean train inference run data \
        diffusion-train diffusion-sample diffusion-full \
        flow-train flow-sample flow-full gif

all: build

build: $(BIN)

$(BIN): $(SRC)
	$(CC) $(CFLAGS) -o $(BIN) $(SRC) $(LIBS)

$(DIFF_BIN): $(DIFF_SRC)
	$(CC) $(CFLAGS) -o $(DIFF_BIN) $(DIFF_SRC) $(LIBS)

$(FLOW_BIN): $(FLOW_SRC)
	$(CC) $(CFLAGS) -o $(FLOW_BIN) $(FLOW_SRC) $(LIBS)

data: $(MNIST)

$(MNIST_GZ):
	curl -L -o $(MNIST_GZ) $(MNIST_URL)

$(MNIST): $(MNIST_GZ)
	gunzip -c $(MNIST_GZ) > $(MNIST)

clean:
	rm -f $(BIN) $(DIFF_BIN) $(FLOW_BIN)

train: build
	./$(BIN)

inference: build
	./$(BIN)

run: train

diffusion-train: $(DIFF_BIN) data
	./$(DIFF_BIN)

diffusion-sample: $(DIFF_BIN)
	./$(DIFF_BIN) --sample

diffusion-full: $(DIFF_BIN) data
	./$(DIFF_BIN)
	python3 make_gif.py --dir output/ddpm

flow-train: $(FLOW_BIN) data
	./$(FLOW_BIN)

flow-sample: $(FLOW_BIN)
	./$(FLOW_BIN) --sample

flow-full: $(FLOW_BIN) data
	./$(FLOW_BIN)
	python3 make_gif.py --dir output/flow

gif:
	python3 make_gif.py
