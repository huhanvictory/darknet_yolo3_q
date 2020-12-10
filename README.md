
# Darknet
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

# Directory analysis:
  HLS kernel: hw/top_kernel_yolov3_int8_16x16.cpp<br>
  OpenCL code: lib_gen/<br>
  Host code: src/network.c<br>

# Run darknet:
### On U250:
  set FPGA=1 SOC=0 TF=0
### On ZCU102:
  set FPGA=1 SOC=1 TF=0

### Commands
  Need Merlin Compiler to generate optimized kernel.
```
  make libgen: to generate opencl library
  make all: to generate host program
  make runsim: generate emulation xclbin by merlin
  make bitgen: generate on board xclbin by merlin
  make run: to run for emulatoin
```

# RUN Tensor Flow:
  set FPGA=1 SOC=1 TF=1
```
  conda info
  conda create --name py37 python=3.7.9
  conda activate py37
  pip install tensorflow==1.15.2
  pip install opencv-python
  python python/tf_demo.py path --batch batch_number --img_path /path/image_in --out_path /path/image_out
  conda deactivate
```

# Current performance:
  U250: 109ms
  ZCU: 170ms

# Todo:
  Support Double Pump DSP
  Solve timing issue
