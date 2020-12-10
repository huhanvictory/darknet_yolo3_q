#ifndef TF_YOLO_OP_H_
#define TF_YOLO_OP_H_

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"

#define NUM_CLASS 20
#define BATCH_SIZE 2

//typedef int DATA_T;

using namespace tensorflow;

REGISTER_OP("My_Yolo_OP")
    .Input("yolo_input: float")
    .Output("yolo_output_s: float")
    .Output("yolo_output_m: float")
    .Output("yolo_output_l: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
    return Status::OK();
    });

class My_Yolo_OP: public OpKernel {
    public:
    explicit My_Yolo_OP(OpKernelConstruction *context);
    ~My_Yolo_OP();
    void Compute(OpKernelContext *context) override;

    private:
    //DATA_T *q_input; 


};

REGISTER_KERNEL_BUILDER(Name("My_Yolo_OP").Device(DEVICE_CPU), My_Yolo_OP);

#endif