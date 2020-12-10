#ifndef __OPENCL_IF_H_INCLUDED__
#define __OPENCL_IF_H_INCLUDED__
#include <stdio.h>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <ap_int.h>
#include "xcl2.hpp"
#include <CL/opencl.h>
#include <CL/cl2.hpp>
#include <CL/cl_ext.h>
#include <CL/cl_ext_xilinx.h>

#include "config_16x16_q.h"
//#include "config_16x16.h"
//#include "config_8x8.h"
//#include "config_4x4.h"
//#include "config.h"
typedef int8_t DATA_T;
typedef int32_t BIAS_DT;

#ifdef __cplusplus
extern "C" {
#endif
int __merlin_init(const char *bitstream);
int __merlin_release();
#ifdef __cplusplus
}
#endif
extern cl::Program* m_program;
extern cl::Context* m_context;
extern cl::CommandQueue* q[OVERLAP];
extern std::vector<DATA_T, aligned_allocator<DATA_T> >  w_in[OVERLAP];
extern std::vector<DATA_T, aligned_allocator<DATA_T> >  data_input[OVERLAP];
extern cl::Kernel *top_kernel;
extern cl_mem_ext_ptr_t ext_buffer_weights[OVERLAP];
extern cl_mem_ext_ptr_t ext_buffer_input[OVERLAP];
extern cl::Buffer *buffer_weights[OVERLAP];
extern cl::Buffer *buffer_input[OVERLAP];
#endif //__OPENCL_IF_H_INCLUDED__
