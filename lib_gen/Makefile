DIE_OPTION=
BANK_OPTION=
#DIE_OPTION=
#BANK_OPTION=
SHELL = /bin/bash

RUNTIME_FLAGS := $(XILINX_SDX)/runtime/include/1_2
RUNTIME_LINK := $(XILINX_SDX)/runtime/lib/x86_64

ifeq ($(TF),1)
TF_CFLAGS ?= $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') 
TF_LFLAGS ?= $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') 
endif

ifeq ($(SOC), 1) 

else
ifdef XILINX_XRT
FILE=$(XILINX_XRT)/include
ifeq ($(FILE), $(wildcard $(FILE)))
RUNTIME_FLAGS := $(FILE)
endif
FILE=$(XILINX_XRT)/lib
ifeq ($(FILE), $(wildcard $(FILE)))
RUNTIME_LINK := $(FILE)
endif
endif

endif

ifdef XILINX_VIVADO
VIVADO_INCLUDE=$(XILINX_VIVADO)/include
endif

ifeq ($(VERBOSE),1)
ECHO :=
else
ECHO := @
endif
ECHO := @

HOST_EXE_DIR = ./
HOST_EXE = host_top

SRCS := $(wildcard *.c *.cpp)
ifeq ($(TF),0)
SRCS := $(filter-out tf_yolo_op.cpp,$(SRCS))
endif

XFLAGS= -fPIC -DHLS_NO_XIL_FPO_LIB \
		-I$(RUNTIME_FLAGS) \
		-I$(VIVADO_INCLUDE) \
		-I$(XILINX_SDX)/Vivado_HLS/include \
		-I$(XILINX_SDX)/include \
		$(TF_CFLAGS) \
		-I../include \
		-I../src \
		-I../
ifeq ($(SOC), 1) 
LINK = -I${SYSROOT}/usr/include/xrt -L${SYSROOT}/usr/lib -lOpenCL -lpthread -lrt -lstdc++ --sysroot=${SYSROOT} -DSOC
else
LINK=   -L$(RUNTIME_LINK) \
		$(TF_LFLAGS) \
	    -lxilinxopencl\
		-pthread -lrt -Wno-unused-label -Wno-narrowing -DVERBOSE -lstdc++  \
		-std=gnu++0x -DLINUX_COMMANDS -fPIC -DIS_PIC \
		-lpthread -ldl 
#		-lpthread -lm -ldl -lrt -fgnu89-inline
		#-lpthread -lm -ldl -lrt -lstdc++ -std=c99 -fgnu89-inline
endif


LICENSE_DIR = ../
LICENSE  = $(LICENSE_DIR)/license/license.o $(LICENSE_DIR)/license/lm_new.o   
LICENSE += -L $(LICENSE_DIR)/license/ -Wl,-rpath=$(LICENSE_DIR)/license/
LICENSE += -std=gnu++0x -DLINUX_COMMANDS -fPIC -DIS_PIC -I $(LICENSE_DIR)/license/
LICENSE += -llmgr_trl -lcrvs -lsb -lnoact -llmgr_dongle_stub -lpthread -ldl

ESTIMATE =
XCLBIN_NAME=kernel_top
XCLBIN = ${XCLBIN_NAME}.xclbin

#supported flow: cpu_sim, hw_sim, hw
ifeq ($(SOC), 1) 
CC = aarch64-linux-gnu-g++
else
CC = g++
endif
#CC = xcpp
CLCC = xocc

CMD = mars_perl
RUN_SCRIPT = $(MERLIN_COMPILER_HOME)/mars-gen/scripts/sdaccel/sdaccel_run_flow.pl

#CLCC_OPT += --kernel ${KERNEL_NAME}

.PHONY: all

all: run

lib_gen:
	@rm -rf bin;
	@mkdir bin;
	$(CC) -O -shared -fPIC -o bin/libkernel.so \
		$(SRCS) $(XFLAGS) $(CFLAGS) $(LINK)

.IGNORE :
	@emconfigutil --platform ${DEVICE} --od . >emconfig.log 2>&1

clean:
	@${RM} -rf ${HOST_EXE} ${XCLBIN} emconfig.json _xocc_${XCLBIN_NAME}_*.dir .Xil bin

cleanall: clean
	@${RM} -rf *.xclbin sdaccel_profile_summary.* _xocc_compile* _xocc_link* _xocc_krnl* TempConfig


