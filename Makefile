MAIN = matMul
NVCC = /opt/cuda/bin/nvcc
BUILD_DIR = ./build
BIN = $(BUILD_DIR)/$(MAIN)


$(shell mkdir -p $(BUILD_DIR))
# SRCS = $(shell find $(abspath ./) -name "*.cu")
INC_PATH ?= 
INCFLAGS = $(addprefix -I, $(INC_PATH))
SRCS +=  $(abspath ./matMul.cu)
LDFLAGS += -lcublas


$(BIN): $(SRCS)
	$(NVCC) $^ $(INCFLAGS) $(LDFLAGS) -o $(abspath $@)


default:$(BIN)

all: default

clean:
	rm -rf $(BUILD_DIR)

.PHONY: default all clean
