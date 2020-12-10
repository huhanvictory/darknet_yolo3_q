#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H
#include "darknet.h"
#include "cuda.h"
#include "math.h"

ACTIVATION get_activation(char *s);

char *get_activation_string(ACTIVATION a);
float activate(float x, ACTIVATION a);
float gradient(float x, ACTIVATION a);
void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta);
void activate_array(float *x, const int n, const ACTIVATION a);
#ifdef GPU
void activate_array_gpu(float *x, int n, ACTIVATION a);
void gradient_array_gpu(float *x, int n, ACTIVATION a, float *delta);
#endif

static inline float stair_activate(float x)
{
    int n = floor(x);
    if (n%2 == 0) return floor(x/2.);
    else return (x - n) + floor(x/2.);
}
static inline float hardtan_activate(float x)
{
    if (x < -1) return -1;
    if (x > 1) return 1;
    return x;
}
static inline float linear_activate(float x){return x;}
static inline float logistic_activate(float x){return 1./(1. + exp(-x));}
static inline float loggy_activate(float x){return 2./(1. + exp(-x)) - 1;}
static inline float relu_activate(float x){return x*(x>0);}
static inline float elu_activate(float x){return (x >= 0)*x + (x < 0)*(exp(x)-1);}
static inline float selu_activate(float x){return (x >= 0)*1.0507*x + (x < 0)*1.0507*1.6732*(exp(x)-1);}
static inline float relie_activate(float x){return (x>0) ? x : .01*x;}
static inline float ramp_activate(float x){return x*(x>0)+.1*x;}
static inline float leaky_activate(float x){return (x>0) ? x : .1*x;}
static inline float tanh_activate(float x){return (exp(2*x)-1)/(exp(2*x)+1);}
static inline float plse_activate(float x)
{
    if(x < -4) return .01 * (x + 4);
    if(x > 4)  return .01 * (x - 4) + 1;
    return .125*x + .5;
}

static inline float lhtan_activate(float x)
{
    if(x < 0) return .001*x;
    if(x > 1) return .001*(x-1) + 1;
    return x;
}
static inline float lhtan_gradient(float x)
{
    if(x > 0 && x < 1) return 1;
    return .001;
}

static inline float hardtan_gradient(float x)
{
    if (x > -1 && x < 1) return 1;
    return 0;
}
static inline float linear_gradient(float x){return 1;}
static inline float logistic_gradient(float x){return (1-x)*x;}
static inline float loggy_gradient(float x)
{
    float y = (x+1.)/2.;
    return 2*(1-y)*y;
}
static inline float stair_gradient(float x)
{
    if (floor(x) == x) return 0;
    return 1;
}
static inline float relu_gradient(float x){return (x>0);}
static inline float elu_gradient(float x){return (x >= 0) + (x < 0)*(x + 1);}
static inline float selu_gradient(float x){return (x >= 0)*1.0507 + (x < 0)*(x + 1.0507*1.6732);}
static inline float relie_gradient(float x){return (x>0) ? 1 : .01;}
static inline float ramp_gradient(float x){return (x>0)+.1;}
static inline float leaky_gradient(float x){return (x>0) ? 1 : .1;}
static inline float tanh_gradient(float x){return 1-x*x;}
static inline float plse_gradient(float x){return (x < 0 || x > 1) ? .01 : .125;}

// static inline int8_t xilinx_quantizer(int32_t input, int divider)
// {
//     double di, fl, rn, ce;
//     di = (double)input / divider;
//     fl = floor(di);
//     rn = round(di);
//     ce = ceil(di);
//     float ret_val;

//     if (input < 0 && (di - fl == 0.5))
//     {
//         ret_val = ce;
//     }
//     else
//     {
//         ret_val = rn;
//     }
//     ret_val = (ret_val < -128.0)? -128.0 : ret_val;
//     ret_val = (ret_val> 127.0)? 127 : ret_val;
//     return ret_val;
// } 

static inline int8_t xilinx_quantizer_shift(int32_t input, int shift_count)
{
    //return xilinx_quantizer(input, pow(2, shift_count));
    int32_t ret_val;
    if (shift_count > 0){
        int right_of_shift = 1 << (shift_count - 1);
        if (input & right_of_shift){
            
            ret_val = (input >> shift_count) + 1;
            //printf("input = %d, shift_count_1=%d, ret_val=%d,", input, shift_count, ret_val);
        }
        else{
            ret_val = input  >> shift_count;
            //printf("input = %d, shift_count_0=%d, ret_val=%d,", input, shift_count, ret_val);
        }
    }
    else{
        ret_val =  input;
    }
    ret_val = (ret_val < -128)? -128.0 : ret_val;
    ret_val = (ret_val > 127)? 127 : ret_val;
    return ret_val;
}

static inline int64_t sum_f(float* in, int len){
    int64_t sum = 0;
    int i_q;
    for (i_q = 0; i_q < len; i_q ++){
        sum = sum + (int64_t)in[i_q];
    }
    return sum;
}
static inline int64_t sum_i(int* in, int len){
    int64_t sum = 0;
    int i_q;
    for (i_q = 0; i_q < len; i_q ++){
        sum = sum + in[i_q];
    }
    return sum;
}
static inline int64_t sum_i8(int8_t* in, int len){
    int64_t sum = 0;
    int i_q;
    for (i_q = 0; i_q < len; i_q ++){
        sum = sum + in[i_q];
    }
    return sum;
}

#endif

