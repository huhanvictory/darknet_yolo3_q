GPU=0
CUDNN=0
OPENCV=0
OPENMP=0
DEBUG=0
FPGA=1
FPGA_SIM=0
DEBUG_CPU=0
DEBUG_FPGA=0
OUTPUT_REF=0
SOC=0
TF=0

ARCH= -gencode arch=compute_30,code=sm_30 \
      -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52]
#      -gencode arch=compute_20,code=[sm_20,sm_21] \ This one is deprecated?

# This is what I use, uncomment if you know your arch and want to specify
# ARCH= -gencode arch=compute_52,code=compute_52

VPATH=./src/:./examples
SLIB=libdarknet.so
ALIB=libdarknet.a
ifeq ($(FPGA), 1) 
OLIB=-L./ -Wl,-rpath=./ -lkernel
else
OLIB=
endif
EXEC=darknet
OBJDIR=./obj/

ifeq ($(SOC), 1) 
CC=aarch64-linux-gnu-gcc
CPP=aarch64-linux-gnu-g++
else
CC=gcc
CPP=g++
endif
NVCC=nvcc 
AR=ar
ARFLAGS=rcs
OPTS=-Ofast
CFLAGS=-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC
ifeq ($(SOC), 1) 
LDFLAGS = -I${SYSROOT}/usr/include/xrt -L${SYSROOT}/usr/lib -lOpenCL -lpthread -lrt -lstdc++ --sysroot=${SYSROOT}
COMMON= -Iinclude/ -Isrc/
COMMON+= -DSOC=1
CFLAGS+= -DSOC=1
else
LDFLAGS= -lm -pthread  -lxilinxopencl -L $(XILINX_XRT)/lib
COMMON= -Iinclude/ -Isrc/ -I $(XILINX_XRT)/include
endif

ifeq ($(OPENMP), 1) 
CFLAGS+= -fopenmp
endif

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)

ifeq ($(OPENCV), 1) 
COMMON+= -DOPENCV
CFLAGS+= -DOPENCV
LDFLAGS+= `pkg-config --libs opencv` -lstdc++
COMMON+= `pkg-config --cflags opencv` 
endif

ifeq ($(FPGA), 1) 
COMMON+= -DFPGA  -O3 -Wno-deprecated-declarations -lm
CFLAGS+= -DFPGA 
LDFLAGS+= -L. -lkernel

VENDOR=XILINX
#DEVICE=vitis::zcu102_base
#DEVICE=vitis::xilinx_zcu102_base_202010_1
#DEVICE=sdaccel::xilinx_u250_xdma_201830_2
DEVICE=vitis::xilinx_u250_xdma_201830_2

KERNEL_NAME=kernel_top
KERNEL_SRC_FILES= ./hw/top_kernel_yolov3_int8_16x16.cpp 

EXE=yolov3_tiny__app
ACC_EXE=$(EXE)_acc
#EXE_ARGS= detect cfg/yolov3_q.cfg yolov3_q.weights data/dog.jpg -thresh  0.2
EXE_ARGS= detect config/Yolov3_q.cfg config/Yolov3_q.weights data/dog.jpg -thresh  0.2

ATTRIBUTE  = -funsafe-math-optimizations
ATTRIBUTE += --attribute coarse_grained_pipeline=off
ATTRIBUTE += --attribute memory_burst=off
ATTRIBUTE += --attribute bus_bitwidth=off
ATTRIBUTE += --attribute memory_coalescing=off
#ATTRIBUTE += --attribute reduction_general=off
#ATTRIBUTE += --attribute reduction_opt=off
#ATTRIBUTE += --attribute line_buffer=off
#ATTRIBUTE += --attribute structural_func_inline=off
#ATTRIBUTE += --attribute function_inline=off
#ATTRIBUTE += --attribute auto_func_inline=off
#ATTRIBUTE += --attribute explicit_bundle=on
ATTRIBUTE += --vendor-options "-g"
ATTRIBUTE += --attribute stream_prefetch=off
ATTRIBUTE += --attribute coarse_grained_parallel=off
ATTRIBUTE += --attribute reduction_general=off

# N16xh:same with python config
# 52 / 104 / 208
N16_LINE:=104
# pingpang buffer size for input burst
# 13 / 26 / 52
# 13: 13 * 16 * 512 * 8bit / 512 bus
# 26: 26 * 28 * 256 * 8bit / 512 bus
# 52: 52 * 52 * 192 * 8bit / 512 bus (192 becausue of one 384 channel layer)
ONCHIP_SIZE:=13
N16_LINE_ATT = -DN16_LINE=$(N16_LINE)
ONCHIP_SIZE_ATT = -DONCHIP_SIZE=$(ONCHIP_SIZE)

CMP_OPT=-d11 -DFPGA   $(ATTRIBUTE) $(ONCHIP_SIZE_ATT) $(N16_LINE_ATT)  -D AP_INT_MAX_W=4096
LNK_OPT=-d11

ifeq ($(FPGA_SIM), 1)
LNK_OPT+= -DFPGA_SIM
CMP_OPT+= -DFPGA_SIM
endif

ifeq ($(DEBUG_CPU), 1) 

DEBUG_LAYER = 1
ifeq ($(DEBUG_LAYER), 1) 
COMMON+= -DDEBUG_CPU
CFLAGS+= -DDEBUG_CPU
endif

DEBUG_DATA_SIM = 0
ifeq ($(DEBUG_DATA_SIM), 1)
COMMON+= -DDEBUG_SIM
CFLAGS+= -DDEBUG_SIM
endif
else
COMMON+= -DCPU
CFLAGS+= -DCPU 
endif


CMP_OPT+= -DDSP_PACK
ifeq ($(DEBUG_FPGA), 1) 
#CMP_OPT+= -DDEBUG_BURST
#CMP_OPT+= -DDEBUG_WEIGHT
#CMP_OPT+= -DDEBUG_CONV
#CMP_OPT+= -DDEBUG_BIAS
#CMP_OPT+= -DDEBUG_SHORTCUT
#CMP_OPT+= -DDEBUG_UPSAMPLE
#CMP_OPT+= -DDEBUG_DATAOUT

COMMON+= -DDEBUG_FPGA
CFLAGS+= -DDEBUG_FPGA
endif

CXX=xcpp
CXX_INC_DIRS= $(COMMON)
CXX_FLAGS += $(LDFLAGS)

endif

ifeq ($(GPU), 1) 
COMMON+= -DGPU -I/usr/local/cuda/include/
CFLAGS+= -DGPU
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
endif

ifeq ($(CUDNN), 1) 
COMMON+= -DCUDNN 
CFLAGS+= -DCUDNN
LDFLAGS+= -lcudnn
endif

OBJ=gemm.o utils.o cuda.o deconvolutional_layer.o convolutional_layer.o list.o image.o activations.o im2col.o col2im.o blas.o crop_layer.o dropout_layer.o maxpool_layer.o softmax_layer.o data.o matrix.o network.o connected_layer.o cost_layer.o parser.o option_list.o detection_layer.o route_layer.o upsample_layer.o box.o normalization_layer.o avgpool_layer.o layer.o local_layer.o shortcut_layer.o logistic_layer.o activation_layer.o rnn_layer.o gru_layer.o crnn_layer.o demo.o batchnorm_layer.o region_layer.o reorg_layer.o tree.o  lstm_layer.o l2norm_layer.o yolo_layer.o iseg_layer.o image_opencv.o
EXECOBJA=captcha.o lsd.o super.o art.o tag.o cifar.o go.o rnn.o segmenter.o regressor.o classifier.o coco.o yolo.o detector.o nightmare.o instance-segmenter.o darknet.o
ifeq ($(GPU), 1) 
LDFLAGS+= -lstdc++ 
OBJ+=convolutional_kernels.o deconvolutional_kernels.o activation_kernels.o im2col_kernels.o col2im_kernels.o blas_kernels.o crop_layer_kernels.o dropout_layer_kernels.o maxpool_layer_kernels.o avgpool_layer_kernels.o
endif

EXECOBJ = $(addprefix $(OBJDIR), $(EXECOBJA))
OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile include/darknet.h

all: obj backup results $(SLIB) $(ALIB) $(EXEC)
#all: obj  results $(SLIB) $(ALIB) $(EXEC)


$(EXEC): $(EXECOBJ) $(ALIB)
	$(CC) -fpermissive -std=c++11 $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(ALIB) $(OLIB)
#	xcpp -fpermissive -std=c++11 $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(ALIB) $(OLIB)

$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

$(SLIB): $(OBJS)
	$(CC) $(CFLAGS) -shared $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CPP) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

obj:
	mkdir -p obj
backup:
	mkdir -p backup
results:
	mkdir -p results

.PHONY: clean

clean:
	rm -rf $(OBJS) $(SLIB) $(ALIB) $(EXEC) $(EXECOBJ) $(OBJDIR)/*

config_gen:
	python3 python/parse_cfg.py --cfg cfg/yolov3_q.cfg --N16xh $(N16_LINE)

runsim:
	python3 python/parse_cfg.py --cfg cfg/yolov3_q.cfg --N16xh $(N16_LINE)
	merlincc -c $(KERNEL_SRC_FILES) -DXILINX -DRUNSIM -o $(KERNEL_NAME) $(CMP_OPT) --platform=$(DEVICE)
	merlincc $(KERNEL_NAME).mco -march=sw_emu -D MCC_SIM -o kernel_top $(LNK_OPT) --platform=$(DEVICE)

estimate:
	merlincc $(KERNEL_NAME).mco --report=estimate -d11 --platform=$(DEVICE)

runhw:
	merlincc $(KERNEL_NAME).mco -march=hw_emu -D MCC_SIM -o kernel_top $(LNK_OPT) --platform=$(DEVICE)
	XCL_EMULATION_MODE=hw_emu ./$(EXEC) $(EXE_ARGS)

bitgen:
	python3 python/parse_cfg.py --cfg cfg/yolov3_q.cfg --N16xh $(N16_LINE)
	merlincc -c $(KERNEL_SRC_FILES) -DXILINX -DBITGEN -o $(KERNEL_NAME) $(CMP_OPT) --platform=$(DEVICE)
	merlincc $(KERNEL_NAME).mco -o kernel_top_hw.xclbin -d11 --platform=$(DEVICE)

libgen:
	rm -rf lib_gen/bin/libkernel.so;
	cd lib_gen; make lib_gen SOC=$(SOC) TF=$(TF); cd -;
	cp lib_gen/bin/libkernel.so .;

runall:
	XCL_EMULATION_MODE=sw_emu ./$(EXEC) $(EXE_ARGS) 

runtest:
	XCL_EMULATION_MODE=sw_emu ./$(EXEC) $(EXE_ARGS) $(START) $(END) 0 16

run0:
	XCL_EMULATION_MODE=sw_emu ./$(EXEC) $(EXE_ARGS) 0 0 0 16

run56:
	XCL_EMULATION_MODE=sw_emu ./$(EXEC) $(EXE_ARGS) 56 56 0 16

run58:
	XCL_EMULATION_MODE=sw_emu ./$(EXEC) $(EXE_ARGS) 58 58 0 16

run59:
	XCL_EMULATION_MODE=sw_emu ./$(EXEC) $(EXE_ARGS) 59 59 0 16

# 13 * 13 * 3
run57:
	XCL_EMULATION_MODE=sw_emu ./$(EXEC) $(EXE_ARGS) 57 57 0 16
# 26 * 26 * 3
run61:
	XCL_EMULATION_MODE=sw_emu ./$(EXEC) $(EXE_ARGS) 61 61 0 16
# 52 * 52 * 3
run69:
	XCL_EMULATION_MODE=sw_emu ./$(EXEC) $(EXE_ARGS) 69 69 0 16
# 104 * 104 * 3
run6:
	XCL_EMULATION_MODE=sw_emu ./$(EXEC) $(EXE_ARGS) 6 6 0 16
# 26 * 26 * 3 - > 13 * 13 * 3
run43:
	XCL_EMULATION_MODE=sw_emu ./$(EXEC) $(EXE_ARGS) 43 43 0 16 > log43
run63:
	XCL_EMULATION_MODE=sw_emu ./$(EXEC) $(EXE_ARGS) 63 63 0 16 > log63
run70:
	XCL_EMULATION_MODE=sw_emu ./$(EXEC) $(EXE_ARGS) 70 70 0 16 > log70

# 0: w=416 size=3 stride=1
# 1: w=416->208 size=3 stride=2
# 4: w=208->104 size=3 stride=2
# 9: w=104->52 size=3 stride=2
# 26: w=52->26 size=3 stride=2
# 43: w=26->13 size=3 stride=2
# 56: w=13 size=1 stride=1
# 57: w=13 size=3 stride=1
# 58: w=13 size=1 stride=1 last 1ayer
# 59: w=13 size=1 stride=1 upsample
# 63: w=26 size=3 stride=1
# 64: w=26 size=1 stride=1
# 69: w=52 size=3 stride=1
# 70: w=52 size=1 stride=1
testall:
	XCL_EMULATION_MODE=sw_emu ./$(EXEC) $(EXE_ARGS) 0   0 0 16 > log00;
	XCL_EMULATION_MODE=sw_emu ./$(EXEC) $(EXE_ARGS) 1   1 0 16 > log01;
	XCL_EMULATION_MODE=sw_emu ./$(EXEC) $(EXE_ARGS) 4   4 0 16 > log04;
	XCL_EMULATION_MODE=sw_emu ./$(EXEC) $(EXE_ARGS) 9   9 0 16 > log09;
	XCL_EMULATION_MODE=sw_emu ./$(EXEC) $(EXE_ARGS) 26 26 0 16 > log26;
	XCL_EMULATION_MODE=sw_emu ./$(EXEC) $(EXE_ARGS) 43 43 0 16 > log43;
	XCL_EMULATION_MODE=sw_emu ./$(EXEC) $(EXE_ARGS) 56 56 0 16 > log56;
	XCL_EMULATION_MODE=sw_emu ./$(EXEC) $(EXE_ARGS) 57 57 0 16 > log57;
	XCL_EMULATION_MODE=sw_emu ./$(EXEC) $(EXE_ARGS) 58 58 0 16 > log58;
	XCL_EMULATION_MODE=sw_emu ./$(EXEC) $(EXE_ARGS) 59 59 0 16 > log59;
	XCL_EMULATION_MODE=sw_emu ./$(EXEC) $(EXE_ARGS) 63 63 0 16 > log63;
	XCL_EMULATION_MODE=sw_emu ./$(EXEC) $(EXE_ARGS) 64 64 0 16 > log64;
	XCL_EMULATION_MODE=sw_emu ./$(EXEC) $(EXE_ARGS) 69 69 0 16 > log69;
	XCL_EMULATION_MODE=sw_emu ./$(EXEC) $(EXE_ARGS) 70 70 0 16 > log70;

detect:
	make
	./$(EXEC) detect cfg/yolov3_q.cfg yolov3_q.weights data/dog.jpg -thresh  0.2 
