#ifndef IM2COL_H
#define IM2COL_H
#include<stdint.h>

void im2col_cpu(int8_t* data_im,
        int channels, int height, int width,
        int ksize, int stride, int pad, int8_t* data_col);

#ifdef GPU

void im2col_gpu(float *im,
         int channels, int height, int width,
         int ksize, int stride, int pad,float *data_col);

#endif
#endif
