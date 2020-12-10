#include "__merlin_opencl_if.h"

cl::Program* m_program;
cl::Context* m_context;
cl::CommandQueue* q[OVERLAP];
cl::Event write_event;
std::vector<DATA_T, aligned_allocator<DATA_T> >   w_in[OVERLAP];
std::vector<DATA_T, aligned_allocator<DATA_T> >   data_input[OVERLAP];
cl::Kernel *top_kernel;
cl_mem_ext_ptr_t ext_buffer_weights[OVERLAP];
cl_mem_ext_ptr_t ext_buffer_input[OVERLAP];
cl::Buffer *buffer_weights[OVERLAP];
cl::Buffer *buffer_input[OVERLAP];

int init(const std::string& binaryFileName) {
    // The get_xil_devices will return vector of Xilinx Devices
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    m_context = new cl::Context(device);
    for(int i=0; i<OVERLAP; i++) {
        q[i] = new cl::CommandQueue(*m_context, device, CL_QUEUE_PROFILING_ENABLE);
    }
    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    std::cout << "Found Device=" << device_name.c_str() << std::endl; 
    auto fileBuf = xcl::read_binary_file(binaryFileName);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    devices.resize(1);
    m_program = new cl::Program(*m_context, devices, bins);
    top_kernel = new cl::Kernel(*m_program, "top_kernel"); 
   
    // double buffer used to do overlap
    for(int i=0; i<OVERLAP; i++) {
#ifdef SOC
        buffer_weights[i] = new cl::Buffer(*m_context, 
                                           CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY,
                                           OUTPUT_LAYER_NUM*1024*1024 + OUTPUT_LAYER_NUM*1024*sizeof(BIAS_DT),
                                           NULL);
        buffer_input[i]   = new cl::Buffer(*m_context, 
                                           CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY,
                                           OUTPUT_LAYER_NUM*1024*1024*sizeof(DATA_T),
                                           NULL);
#else
        // set size for extend buffer
        // data_input include all layer input and output
        w_in[i].resize(OUTPUT_LAYER_NUM*1024*1024 + OUTPUT_LAYER_NUM*1024*sizeof(BIAS_DT)); 
        data_input[i].resize(OUTPUT_LAYER_NUM*1024*1024);
   
        // map host bufer to opencl buffer
        ext_buffer_weights[i].obj       = w_in[i].data();
        ext_buffer_weights[i].param     = 0;
        ext_buffer_weights[i].flags     = XCL_MEM_DDR_BANK0;
        ext_buffer_input[i].obj         = data_input[i].data();
        ext_buffer_input[i].param       = 0;
        ext_buffer_input[i].flags       = XCL_MEM_DDR_BANK0;

        // create opencl buffer
        buffer_weights[i] = new cl::Buffer(*m_context, 
                                           CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                           (OUTPUT_LAYER_NUM*1024*1024 + OUTPUT_LAYER_NUM*1024*sizeof(BIAS_DT)),
                                           &ext_buffer_weights[i]);     
        buffer_input[i]   = new cl::Buffer(*m_context,
                                           CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                           OUTPUT_LAYER_NUM*1024*1024*sizeof(DATA_T),
                                           &ext_buffer_input[i]);
#endif
    }
    return 0;
}
int release() {
    delete (top_kernel);
    delete (m_program);
    delete (q[0]);
    delete (q[1]);
    delete (m_context);
    return 0;
}

int __merlin_init(const char *bitstream) {
  init(bitstream);
  return CL_SUCCESS;
}

int __merlin_release() {
  release();
  return CL_SUCCESS;
}
