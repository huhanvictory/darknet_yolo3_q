#include "shortcut_layer.h"
#include "cuda.h"
#include "blas.h"
#include "activations.h"

#include <stdio.h>
#include <assert.h>

layer make_shortcut_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2)
{
    fprintf(stderr, "res  %3d                %4d x%4d x%4d   ->  %4d x%4d x%4d\n",index, w2,h2,c2, w,h,c);
    layer l = {0};
    l.type = SHORTCUT;
    l.batch = batch;
    l.w = w2;
    l.h = h2;
    l.c = c2;
    l.out_w = w;
    l.out_h = h;
    l.out_c = c;
    l.outputs = w*h*c;
    l.inputs = l.outputs;

    l.index = index;

    l.delta =  calloc(l.outputs*batch, sizeof(float));
    l.output = calloc(l.outputs*batch, sizeof(float));;

    l.forward = forward_shortcut_layer;
    l.backward = backward_shortcut_layer;
    #ifdef GPU
    l.forward_gpu = forward_shortcut_layer_gpu;
    l.backward_gpu = backward_shortcut_layer_gpu;

    l.delta_gpu =  cuda_make_array(l.delta, l.outputs*batch);
    l.output_gpu = cuda_make_array(l.output, l.outputs*batch);
    #endif
    return l;
}

void resize_shortcut_layer(layer *l, int w, int h)
{
    assert(l->w == l->out_w);
    assert(l->h == l->out_h);
    l->w = l->out_w = w;
    l->h = l->out_h = h;
    l->outputs = w*h*l->out_c;
    l->inputs = l->outputs;
    l->delta =  realloc(l->delta, l->outputs*l->batch*sizeof(float));
    l->output = realloc(l->output, l->outputs*l->batch*sizeof(float));

#ifdef GPU
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->output_gpu  = cuda_make_array(l->output, l->outputs*l->batch);
    l->delta_gpu   = cuda_make_array(l->delta,  l->outputs*l->batch);
#endif
    
}


void forward_shortcut_layer(const layer l, network net)
{
    static int shortcut_index = 0;

    shortcut_index += 1;

    int32_t* near_output = calloc(l.outputs, sizeof(int32_t));  // we need to use int32 here as input1 or input2 need to shift left (based on the difference between ipos1 and ipos2) the sum together
    int32_t* far_output = calloc(net.layers[l.index].outputs, sizeof(int32_t));
    int max_ipos = (l.ipos1 > l.ipos2) ? l.ipos1 : l.ipos2;

    //int32_t near_multiplier = pow(2, max_ipos - l.ipos1);
    int i_q;
    for (i_q = 0; i_q < l.outputs; ++i_q)
        near_output[i_q] = net.input[i_q] << (max_ipos - l.ipos1);
        
    //copy_cpu(l.outputs*l.batch, near_output, 1, l.output, 1);  we need to put result to the output after adjusting the range with output opos

    int sum_near = sum_f(l.output, l.outputs);

    //int far_multiplier = pow(2, max_ipos - l.ipos2);
    for (i_q = 0; i_q < net.layers[l.index].outputs; ++i_q)
        far_output[i_q] = net.layers[l.index].output[i_q] << (max_ipos - l.ipos2);

    int sum_far = sum_f(far_output, l.outputs);

    shortcut_cpu(l.batch, l.w, l.h, l.c, far_output, l.out_w, l.out_h, l.out_c, l.alpha, l.beta, near_output);
    // activate_array(l.output, l.outputs*l.batch, l.activation); In yolo V3 all the short cut layers have linear activation so we can comment this line. 

    //int divider = pow(2, max_ipos - l.opos);
    for (i_q = 0; i_q < l.outputs; ++i_q){
        //l.output[i_q] = xilinx_quantizer(near_output[i_q], divider);
        l.output[i_q] = xilinx_quantizer_shift(near_output[i_q], max_ipos - l.opos);
    }
    int sum_act = sum_f(l.output, l.outputs);
}

void backward_shortcut_layer(const layer l, network net)
{
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
    axpy_cpu(l.outputs*l.batch, l.alpha, l.delta, 1, net.delta, 1);
    shortcut_cpu(l.batch, l.out_w, l.out_h, l.out_c, l.delta, l.w, l.h, l.c, 1, l.beta, net.layers[l.index].delta);
}

#ifdef GPU
void forward_shortcut_layer_gpu(const layer l, network net)
{
    copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
    shortcut_gpu(l.batch, l.w, l.h, l.c, net.layers[l.index].output_gpu, l.out_w, l.out_h, l.out_c, l.alpha, l.beta, l.output_gpu);
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

void backward_shortcut_layer_gpu(const layer l, network net)
{
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    axpy_gpu(l.outputs*l.batch, l.alpha, l.delta_gpu, 1, net.delta_gpu, 1);
    shortcut_gpu(l.batch, l.out_w, l.out_h, l.out_c, l.delta_gpu, l.w, l.h, l.c, 1, l.beta, net.layers[l.index].delta_gpu);
}
#endif
