#include <stdlib.h>
#include <stdio.h>
#include "sys/time.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tf_yolo_op.h"
#include "../src/activations.h"
#include "../include/darknet.h"
#include "../src/config_16x16_q.h"
#include "__merlinhead_kernel_top.h"

double what_time_is_it_now()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
void convert_inp_to_darknet(const Tensor &src, Tensor &dst) // A tensor in TF is in NHWC format but Darknet keeps them in NCHW and also do the first layer multiplication by 64
{
  auto src_map = src.tensor<float, 4>();
  auto dst_map = dst.tensor<int8_t, 4>();
  const TensorShape &src_shape = src.shape();
  int N, H, W, C;
  N = src_shape.dim_size(0);
  C = src_shape.dim_size(1);
  H = src_shape.dim_size(2);
  W = src_shape.dim_size(3);

  //std::cout << N << " X " << C << " X " << H << " X " << W << "\n";

  for (int n = 0; n < N; ++n)
  {
    for (int c = 0; c < C; ++c)
    {
      for (int h = 0; h < H; ++h)
      {
        for (int w = 0; w < W; ++w)
        {
          dst_map(n, c, h, w) = xilinx_quantizer_shift(round(src_map(n, c, h, w) * 64), 0);
        }
      }
    }
  }
}
void convert_from_darknet(const Tensor &src, Tensor &dst)
{
  auto src_map = src.tensor<float, 4>();
  auto dst_map = dst.tensor<float, 4>();
  const TensorShape &src_shape = src.shape();
  int N, H, W, C;
  N = src_shape.dim_size(0);
  C = src_shape.dim_size(1);
  H = src_shape.dim_size(2);
  W = src_shape.dim_size(3);

  for (int n = 0; n < N; ++n)
  {
    for (int c = 0; c < C; ++c)
    {
      for (int h = 0; h < H; ++h)
      {
        for (int w = 0; w < W; ++w)
        {
          dst_map(n, h, w, c) = src_map(n, c, h, w);
        }
      }
    }
  }
}
My_Yolo_OP::My_Yolo_OP(OpKernelConstruction *context) : OpKernel(context)
{
  double time = what_time_is_it_now();
  __merlin_init("/curr/reza/Projects/YOLOv3_quantization/config/kernel_top.xclbin");  // initializing the accelerator
  const char *bias_path = "/curr/reza/Projects/YOLOv3_quantization/config/raw_bias.bin";
  const char *weight_path = "/curr/reza/Projects/YOLOv3_quantization/config/raw_wgts.bin";

  // Loading biases in int32
  int32_t bias_in[OUTPUT_LAYER_NUM][1024] = {0};
  FILE *fp = fopen(bias_path, "rb");
  for (int l_cnt = 0; l_cnt < OUTPUT_LAYER_NUM; ++l_cnt)
  {
    fread(bias_in[l_cnt], sizeof(int32_t), 1024, fp);
  }
  fclose(fp);

  //loading weights in int8
  DATA_T *weights_in[OUTPUT_LAYER_NUM];
  fp = fopen(weight_path, "rb");
  for (int l_cnt = 0; l_cnt < OUTPUT_LAYER_NUM; ++l_cnt)
  {
    int weight_len = -1;
    if (l_cnt == 0)
    {
      weight_len = config_list_all[0][0][0] * config_list_all[0][0][0] * PARALLEL_FILTER * config_list_all[0][0][7];
    }
    else
    {
      weight_len = config_list_all[l_cnt][0][0] * config_list_all[l_cnt][0][0] * config_list_all[l_cnt][0][3] * config_list_all[l_cnt][0][7];
      
    }
    weights_in[l_cnt] = (DATA_T*)malloc(weight_len * sizeof(DATA_T));
    fread(weights_in[l_cnt], sizeof(DATA_T), weight_len, fp);
  }
  fclose(fp);

  __merlin_load_weight(weights_in, bias_in);
  for (int l_cnt =0; l_cnt < OUTPUT_LAYER_NUM; ++l_cnt){
    free(weights_in[l_cnt]);
  }

  std::cout << "Loading weights to FPGA and initialization took " << what_time_is_it_now() - time << " seconds!\n";
}

My_Yolo_OP::~My_Yolo_OP(){
  __merlin_release();
  std::cout << "FPGA released!\n";
}

void My_Yolo_OP::Compute(OpKernelContext *context)
{

  const Tensor &input = context->input(0);        // the input tensor
  const TensorShape &input_shape = input.shape(); // the input shape
  int batch_size = input_shape.dim_size(0);
  Tensor *output_0 = NULL;
  OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({batch_size, 3 * (NUM_CLASS + 5), 13, 13}),
                                                   &output_0));
  Tensor *output_1 = NULL;
  OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({batch_size, 3 * (NUM_CLASS + 5), 26, 26}),
                                                   &output_1));
  Tensor *output_2 = NULL;
  OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({batch_size, 3 * (NUM_CLASS + 5), 52, 52}),
                                                   &output_2));

  auto q_input = Tensor(tensorflow::DT_INT8, TensorShape({batch_size, input_shape.dim_size(1), input_shape.dim_size(2), input_shape.dim_size(3)}));

  convert_inp_to_darknet(input, q_input); // quantizing input

  int debug_config[10] = {0, 0, 0, 74, 416, 3};
  double kernel_start = what_time_is_it_now();  
  __merlin_exec_top_kernel_overlap((DATA_T *)q_input.flat<DATA_T>().data(), (float *)output_0->flat<float>().data(),
  (float *)output_1->flat<float>().data(), (float *)output_2->flat<float>().data(), batch_size, debug_config);
  double kernel_end = what_time_is_it_now();
  //std::cout << "Kernel execution took " << batch_size / (what_time_is_it_now() - kernel_start) << " iamges per second! (" << kernel_end - kernel_start <<" seconds) \n";
}
