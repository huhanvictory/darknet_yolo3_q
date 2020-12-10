#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "../src/config_16x16_q.h"

typedef ap_int< ORG_DATA_WIDTH > IMAGE_DT;
typedef float BIAS_DT;
typedef ap_int< ORG_DATA_WIDTH*2 > IMAGE_2DT;
typedef ap_int< ORG_DATA_WIDTH*4 > IMAGE_4DT;
typedef ap_int< ORG_DATA_WIDTH*8 > IMAGE_8DT;
typedef ap_int< ORG_DATA_WIDTH*4 > CONV_DT;

template <typename To, typename From>
inline To Reinterpret(const From& val){
    #pragma HLS inline
    return reinterpret_cast<const To&>(val);
}

IMAGE_DT xilinx_quantizer_shift(IMAGE_4DT input, int shift_count)
{
    IMAGE_4DT ret_val;
    if (shift_count > 0){
        int right_of_shift = 1 << (shift_count - 1);
        if (input & right_of_shift){
            
            ret_val = (input >> shift_count) + 1;
        }
        else{
            ret_val = (input >> shift_count);
        }
    }
    else{
        ret_val =  input;
    }
    if(ret_val > 127)
        ret_val = 127;
    else if(ret_val < -128)
        ret_val = -128;
    return ret_val;
}

void stream_in_weights(
    ap_int< WIDE_BUS_WIDTH > *in_weights,
    hls::stream< ap_int< WIDE_BUS_WIDTH > > & fifo_weights_3x3,
    hls::stream< ap_int< WIDE_BUS_WIDTH > > & fifo_weights_1x1,
    hls::stream< ap_int< WIDE_BUS_WIDTH > > & fifo_bias,
    const int config_list[32]) 
{
    #pragma HLS inline off
    int conv_en = config_list[0];
    int size = config_list[0];
    //int in_w = config_list[1];
    //int in_h = config_list[2];
    int in_c = config_list[3];
    int n = config_list[7];
    int burst_length_filter = config_list[22];
    int in_h_13 = config_list[27];
    int w_offset = config_list[30];
    int bias_offset = config_list[31];
    int trans_cnt = n / PARALLEL_FILTER;
#ifdef DEBUG_WEIGHT
    printf("start stream in weights \n");
    int in_w = config_list[1];
    int in_h = config_list[2];
    printf("in_w:%d, in_h:%d, in_c:%d ", in_w, in_h, in_c);
    printf("n:%d trans_cnt:%d\n", n, trans_cnt);
    printf("w_offset:%d, bias_offset:%d\n",w_offset,bias_offset);
#endif
    int burst_length_single = PARALLEL_FILTER * size * size * PARALLEL_FILTER * ORG_DATA_WIDTH / WIDE_BUS_WIDTH;
    //burst_length_filter = in_c * size * size * PARALLEL_FILTER * ORG_DATA_WIDTH / WIDE_BUS_WIDTH;
    ap_int< WIDE_BUS_WIDTH > buf_in_weights[1152];
    ap_int< WIDE_BUS_WIDTH > buf_in_bias[64];
    //weights data input conv 3x3
    memcpy((void*)buf_in_bias, (void*)&in_weights[bias_offset], sizeof(ap_int< WIDE_BUS_WIDTH >) * 64);
    for (int k = 0; k < trans_cnt; k++) {
        #pragma HLS loop_tripcount min=2 max=2
        ap_int< WIDE_BUS_WIDTH > bias_tmp = buf_in_bias[k];
        fifo_bias . write(bias_tmp);
#ifdef DEBUG_WEIGHT
        for (int p = 0; p < 16; p++) 
        {
            IMAGE_4DT bias = bias_tmp(((p + 1) * 32 - 1),(p * 32));
            printf("load_biases[%d]:%d\n", p, (bias . to_int()));
        }
#endif
        memcpy((void*)buf_in_weights, (void*)&in_weights[k * burst_length_filter + w_offset], sizeof(ap_int< WIDE_BUS_WIDTH >) * burst_length_filter);
        for (int z = 0; z < in_h_13; z++) {
            #pragma HLS loop_tripcount min = 32 max =32
            for (int s = 0; s < in_c / PARALLEL_FILTER; s++) {
                #pragma HLS loop_tripcount min = 1 max =1
                //printf("k:%d/%d, z:%d/%d, s:%d/%d\n",k + 1,trans_cnt,z + 1,in_h / 13,s + 1,in_c / 16);
                for (int j = 0; j < burst_length_single; j++) {
                    #pragma HLS pipeline
                    #pragma HLS loop_tripcount min=36 max=36
                    ap_int< WIDE_BUS_WIDTH > buf_input = buf_in_weights[burst_length_single * s + j];
                    if(conv_en == 1)
                        fifo_weights_1x1.write(buf_input);
                    else
                        fifo_weights_3x3.write(buf_input);
#ifdef DEBUG_WEIGHT
                    for (int f = 0; f < FACTORS; f++) {
                        for (int p = 0; p < 16; p++){
                            ap_int< ORG_DATA_WIDTH > tmp_x;
                            tmp_x = buf_input((f * 128 + p*8 + 7),(f * 128 + p*8));
                            printf("debug_weight[%3d][%3d]=%3d ", p, s*burst_length_single*4+j*4+f,(tmp_x . to_int()));
                        }
                        printf("\n");
                    }
#endif
                }
            }
        }
    }
#ifdef DEBUG_WEIGHT
    printf("finish stream in weights \n");
#endif
}

void write_fifo(
    ap_int< WIDE_BUS_WIDTH > *in_image,
    int write_fifo_line,
    bool &flag_fifo,
    int in_w,
    int conv_en,
    hls::stream< ap_int< WIDE_BUS_WIDTH > > &fifo_image_1x1,
    hls::stream< ap_int< WIDE_BUS_WIDTH > > &fifo_image_ping,
    hls::stream< ap_int< WIDE_BUS_WIDTH > > &fifo_image_pang) 
{
    merlinL15:
    for (int i = 0; i < write_fifo_line; i++) {
        #pragma HLS loop_tripcount min=15 max=15
        merlinL14:
        for (int j = 0; j < in_w / FACTORS; j++) {
            #pragma HLS loop_tripcount min=104 max=104
            if (j == 0) {
                flag_fifo  = !flag_fifo;
            }
            ap_int< WIDE_BUS_WIDTH > buf_input = in_image[i * in_w / FACTORS + j];
            if (conv_en == 1) {
                fifo_image_1x1 . write(buf_input);
#ifdef DEBUG_BURST
                for(int f = 0; f < FACTORS; f++){
                    ap_int< ORG_DATA_WIDTH > tmp_x = (buf_input(((f*PARALLEL_FILTER + 1) * ORG_DATA_WIDTH - 1), (f*PARALLEL_FILTER * ORG_DATA_WIDTH)));
                    printf("conv1x1_data[%3d][%3d]=%3d ", i,j*4+f,(tmp_x . to_int()));
                }
#endif
            }
            else {
                if (flag_fifo) {
                    fifo_image_ping . write(buf_input);
#ifdef DEBUG_BURST
                    for(int f = 0; f < FACTORS; f++){
                        ap_int< ORG_DATA_WIDTH > tmp_x = (buf_input(((f*PARALLEL_FILTER + 1) * ORG_DATA_WIDTH - 1), (f*PARALLEL_FILTER * ORG_DATA_WIDTH)));
                        printf("conv3x3_ping_data[%3d][%3d]=%3d ", i,j*4+f,(tmp_x . to_int()));
                    }
#endif
                }
                else {
                    fifo_image_pang . write(buf_input);
#ifdef DEBUG_BURST
                    for(int f = 0; f < FACTORS; f++){
                        ap_int< ORG_DATA_WIDTH > tmp_x = (buf_input(((f*PARALLEL_FILTER + 1) * ORG_DATA_WIDTH - 1), (f*PARALLEL_FILTER * ORG_DATA_WIDTH)));
                        printf("conv3x3_pang_data[%3d][%3d]=%3d ", i,j*4+f,(tmp_x . to_int()));
                    }
#endif
                }
            }
        }
#ifdef DEBUG_BURST
        printf("\n");
#endif
    }
}


void stream_in_image(
    ap_int< WIDE_BUS_WIDTH > *in_image,
    hls::stream< ap_int< WIDE_BUS_WIDTH > > &fifo_image_ping,
    hls::stream< ap_int< WIDE_BUS_WIDTH > > &fifo_image_pang,
    hls::stream< ap_int< WIDE_BUS_WIDTH > > &fifo_image_1x1,
    const int config_list[32])
{
    #pragma HLS inline off
    int conv_en = config_list[0];
    //int size = config_list[0];
    int in_w = config_list[1];
    int in_h = config_list[2];
    int in_c = config_list[3];
    int n = config_list[7];
    int stride = config_list[9];
    int pad = config_list[10];
    int in_wh = config_list[16];
    int split_h = config_list[20];
    int burst_length = config_list[24];
    //int in_w_13 = config_list[26];//in_w*SPLITING_FACTOR
    int in_h_13 = config_list[27];//in_h/SPLITING_FACTOR
    int input_idx = config_list[28];
    int trans_cnt = n / PARALLEL_FILTER;
#ifdef DEBUG_BURST
    printf("start stream in image\n");
    printf("in_w:%d, in_h:%d, in_c:%d ",in_w,in_h,in_c);
    printf("pad:%d stride:%d ", pad,stride);
    printf("n:%d trans_cnt:%d\n", n,trans_cnt);
    printf("input_idx:%d\n", input_idx);
#endif
    bool flag_bram = 0;
    bool flag_fifo = 0;
    #if ONCHIP_SIZE == 13
    bool flag_onchip = (in_h == 13);
    class ap_int< WIDE_BUS_WIDTH > buf_ping_in_image[13*16/2*1024*8/512];
    class ap_int< WIDE_BUS_WIDTH > buf_pang_in_image[13*16/2*1024*8/512];
    #elif ONCHIP_SIZE == 26
    bool flag_onchip = (in_h == 13 || in_h == 26);
    class ap_int< WIDE_BUS_WIDTH > buf_ping_in_image[26*28/2*512*8/512];
    class ap_int< WIDE_BUS_WIDTH > buf_pang_in_image[26*28/2*512*8/512];
    #elif ONCHIP_SIZE == 52
    bool flag_onchip = (in_h == 13 || in_h == 26 || in_h == 52);
    class ap_int< WIDE_BUS_WIDTH > buf_ping_in_image[52*52/2*384*8/512];
    class ap_int< WIDE_BUS_WIDTH > buf_pang_in_image[52*52/2*384*8/512];
    #else
    ap_int< WIDE_BUS_WIDTH > buf_ping_in_image[(SPLITING_FACTOR+2)*416*PARALLEL_FILTER*ORG_DATA_WIDTH/WIDE_BUS_WIDTH];
    ap_int< WIDE_BUS_WIDTH > buf_pang_in_image[(SPLITING_FACTOR+2)*416*PARALLEL_FILTER*ORG_DATA_WIDTH/WIDE_BUS_WIDTH];
    #endif
    #if N16_LINE == 52
    int N16xh = 52;//16x4
    #elif N16_LINE == 104
    int N16xh = 104;//16x8
    #elif N16_LINE == 208
    int N16xh = 208;//16x16
    #endif
    //printf("stream_in_image_0:burst_length = %d\n", burst_length);
    //burst_length = in_w * PARALLEL_FILTER * (split_h + 2 * pad) * ORG_DATA_WIDTH / WIDE_BUS_WIDTH;
    //printf("stream_in_image_1:burst_length = %d\n", burst_length);
    int data_left = in_h % (split_h + stride - 1);

    int ddr_index = 0;
    int bram_ping_index = 0;
    int bram_pang_index = 0;
    memcpy(&buf_ping_in_image[bram_ping_index], &in_image[input_idx], sizeof(ap_int< WIDE_BUS_WIDTH >) * burst_length);
    merlinL9:
    for (int t = 0; t < trans_cnt; t++) {
        //printf("t:%d/%d ", t + 1,trans_cnt);
        #pragma HLS loop_tripcount min = 2 max = 2
        merlinL8:
        for (int z = 0; z < in_h_13; z++) {
            //printf("z:%d/%d ", z + 1,in_h / 13);
            #pragma HLS loop_tripcount min = 32 max =32
            merlinL7:
            for (int s = 0; s < in_c / PARALLEL_FILTER; s++) {
                #pragma HLS loop_tripcount min = 2 max = 2
#ifdef DEBUG_BURST
               printf("\nt:%d/%d z:%d/%d s:%d/%d, flag_bram:%d\n",t,trans_cnt,z,in_h / split_h,s,in_c / 16,flag_bram);
#endif
                flag_bram = !flag_bram;
                int write_fifo_line;
                if (z == 0) {
                    if (in_h == split_h) {
                        write_fifo_line = N16xh;
                        //write_fifo_line = split_h;
                    }
                    else {
                        write_fifo_line = split_h + pad;
                    }
                }
                else {
                    if (z == in_h_13 - 1) {
                        if (data_left == 0 && stride == 1) {
                                write_fifo_line = split_h + pad;
                        } else {
                            write_fifo_line = data_left + pad;
                        }
                    } else {
                        write_fifo_line = split_h + pad * 2;
                    }
                }
#ifdef DEBUG_BURST
                printf("s1:%d ddr_index0=%d ", s, ddr_index);
#endif
                if(in_h == split_h){
                    int N16 = N16xh / in_h;
                    if (s + N16 == in_c / PARALLEL_FILTER) {
                        ddr_index = 0;
                    } else {
                        ddr_index = (s / N16 + 1) * in_wh * N16 ;
                    }
                    s = s + N16 - 1;
#ifdef DEBUG_BURST
                printf("N16:%d ", N16);
#endif
                } else {
                    if (s + 1 == in_c / PARALLEL_FILTER) {
                        if (z + 1 == in_h_13) {
                            ddr_index = 0;
                        } else {
                            ddr_index = ((z + 1) * (split_h + stride - 1) - pad) * in_w;
                        }
                    } else {
                        if (z == 0) {
                            ddr_index = (s + 1) * in_wh;
                        } else {
                            ddr_index = (s + 1) * in_wh + (z * (split_h + stride - 1) - pad) * in_w;
                        }
                    }
                }
                ddr_index = ddr_index / FACTORS;
#ifdef DEBUG_BURST
                printf(" s2:%d ddr_index1=%d, in_h:%d,split_h=%d, write_fifo_line:%d, burst_length:%d\n",s,ddr_index,in_h,split_h,write_fifo_line,burst_length);
#endif
                if (flag_bram == 1) {
                    if(flag_onchip == 1){
                        if(t == 0 && s < in_c / 16 - 1){
#ifdef DEBUG_BURST
                            printf("copy to pang, read ping, write_fifo_line:%d ", write_fifo_line);
                            printf("bram_ping_index:%d, bram_pang_index:%d\n", bram_ping_index, bram_pang_index);
#endif
                            memcpy(&buf_pang_in_image[bram_pang_index], &in_image[ddr_index + input_idx], sizeof(ap_int< WIDE_BUS_WIDTH >) * burst_length);
                            write_fifo(&buf_ping_in_image[bram_ping_index], write_fifo_line, flag_fifo, in_w,conv_en, fifo_image_1x1, fifo_image_ping, fifo_image_pang);
                        } else {
#ifdef DEBUG_BURST
                            printf("only read ping, write_fifo_line:%d ", write_fifo_line);
                            printf("bram_ping_index:%d, bram_pang_index:%d\n", bram_ping_index, bram_pang_index);
#endif
                            write_fifo(&buf_ping_in_image[bram_ping_index], write_fifo_line, flag_fifo,in_w, conv_en, fifo_image_1x1, fifo_image_ping, fifo_image_pang);
                        }
                        if(z + 1 == in_h_13 && s + 1 == in_c / 16){
                            bram_ping_index = 0;
                            bram_pang_index = 0;
                        } else {
                            bram_ping_index += write_fifo_line * in_w / FACTORS;
                        }
                    } else {
                        bram_ping_index = 0;
                        if (t + 1 == trans_cnt && z + 1 == in_h_13 && s + 1 == in_c / 16) {
#ifdef DEBUG_BURST
                            printf("only read ping, write_fifo_line:%d ", write_fifo_line);
                            printf("bram_ping_index:%d, bram_pang_index:%d\n", bram_ping_index, bram_pang_index);
#endif
                            write_fifo(&buf_ping_in_image[bram_ping_index], write_fifo_line, flag_fifo, in_w, conv_en, fifo_image_1x1, fifo_image_ping, fifo_image_pang);
                        }
                        else {
#ifdef DEBUG_BURST
                            printf("copy to pang, read ping, write_fifo_line:%d ", write_fifo_line);
                            printf("bram_ping_index:%d, bram_pang_index:%d\n", bram_ping_index, bram_pang_index);
#endif
                            memcpy(&buf_pang_in_image[bram_pang_index], &in_image[ddr_index + input_idx], sizeof(ap_int< WIDE_BUS_WIDTH >) * burst_length);
                            write_fifo(&buf_ping_in_image[bram_ping_index], write_fifo_line, flag_fifo, in_w, conv_en, fifo_image_1x1, fifo_image_ping, fifo_image_pang);
                        }
                    }
                } else {
                    if(flag_onchip == 1){
                        if(t == 0 && s < in_c / 16 - 1){
#ifdef DEBUG_BURST
                            printf("copy to ping, read pang, write_fifo_line:%d ", write_fifo_line);
                            printf("bram_ping_index:%d, bram_pang_index:%d\n", bram_ping_index, bram_pang_index);
#endif
                            memcpy(&buf_ping_in_image[bram_ping_index], &in_image[ddr_index + input_idx], sizeof(ap_int< WIDE_BUS_WIDTH >) * burst_length);
                            write_fifo(&buf_pang_in_image[bram_pang_index], write_fifo_line, flag_fifo, in_w, conv_en, fifo_image_1x1, fifo_image_ping, fifo_image_pang);
                        } else {
#ifdef DEBUG_BURST
                            printf("only read pang, write_fifo_line:%d ", write_fifo_line);
                            printf("bram_ping_index:%d, bram_pang_index:%d\n", bram_ping_index, bram_pang_index);
#endif
                            write_fifo(&buf_pang_in_image[bram_pang_index],write_fifo_line,flag_fifo,in_w,conv_en,fifo_image_1x1,fifo_image_ping,fifo_image_pang);
                        }
                        if(z + 1 == in_h_13 && s + 1 == in_c / 16){
                            bram_ping_index = 0;
                            bram_pang_index = 0;
                        } else {
                            bram_pang_index += write_fifo_line * in_w / FACTORS;
                        }
                    } else {
                        bram_pang_index = 0;
                        if (t + 1 == trans_cnt && z + 1 == in_h_13 && s + 1 == in_c / 16) {
#ifdef DEBUG_BURST
                            printf("only read pang, write_fifo_line:%d ", write_fifo_line);
                            printf("bram_ping_index:%d, bram_pang_index:%d\n", bram_ping_index, bram_pang_index);
#endif
                            write_fifo(&buf_pang_in_image[bram_pang_index], write_fifo_line, flag_fifo, in_w, conv_en, fifo_image_1x1, fifo_image_ping, fifo_image_pang);
                        }
                        else {
#ifdef DEBUG_BURST
                            printf("copy to ping, read pang, write_fifo_line:%d ", write_fifo_line);
                            printf("bram_ping_index:%d, bram_pang_index:%d\n", bram_ping_index, bram_pang_index);
#endif
                            memcpy(&buf_ping_in_image[bram_ping_index], &in_image[ddr_index + input_idx], sizeof(ap_int< WIDE_BUS_WIDTH >) * burst_length);
                            write_fifo(&buf_pang_in_image[bram_pang_index], write_fifo_line, flag_fifo, in_w, conv_en, fifo_image_1x1, fifo_image_ping, fifo_image_pang);
                        }
                    }
                }
            }
        }
    }
#ifdef DEBUG_BURST
    printf("finish stream in image\n");
#endif
}


void conv_1x1_core(
    hls::stream< ap_int< WIDE_BUS_WIDTH > > &stream_data_in,
#ifdef DSP_PACK
    IMAGE_2DT weights_in[PARALLEL_FILTER/2][TILING_IMAGE],
    IMAGE_8DT sum_buffer[PARALLEL_FILTER/2][2*SPLITING_FACTOR*208],
#else
    IMAGE_DT weights_in[PARALLEL_FILTER][TILING_IMAGE],
    IMAGE_4DT sum_buffer[PARALLEL_FILTER][2*SPLITING_FACTOR*208],
#endif
    int one_compute_iter,
    bool init,
    bool end) 
{
    #pragma HLS INLINE
#ifdef DEBUG_BURST
    printf("start_conv_1x1_core_1\n");
#endif
    ap_int< WIDE_BUS_WIDTH > buf_input;
    #pragma ACCEL pipeline flatten
    for (int j = 0; j < one_compute_iter / FACTORS; ++j) {
        #pragma HLS loop_tripcount min=676 max=676
        for (int f = 0; f < FACTORS; f++) {
            if(f == 0) {
                buf_input = (stream_data_in . read());
            }
            #ifdef DSP_PACK
            for (int p = 0; p < PARALLEL_FILTER / 2; p++) { 
                IMAGE_4DT result_sum0 = 0;
                IMAGE_4DT result_sum1 = 0;
            #else
            for (int p = 0; p < PARALLEL_FILTER; p++) { 
                IMAGE_4DT result_sum = 0;
            #endif
                for (int l = 0; l < TILING_IMAGE; l++) {
                    #ifdef DSP_PACK
                    IMAGE_DT image_tmp = (buf_input(((f*PARALLEL_FILTER + l + 1) * ORG_DATA_WIDTH - 1), ((f*PARALLEL_FILTER + l) * ORG_DATA_WIDTH)));
                    IMAGE_DT weight0_tmp = weights_in[p][l](7,0);
                    IMAGE_DT weight1_tmp = weights_in[p][l](15,8);
                    ap_int<27> w_tmp0 = 0;
                    w_tmp0(7,0)= weight0_tmp;
                    w_tmp0(26,8) = (weight0_tmp(7,7) == 1) ? 0x7ffff : 0;
                    //w_tmp0(26,8) = (weight0_tmp(7,7),weight0_tmp(7,7),weight0_tmp(7,7),weight0_tmp(7,7),weight0_tmp(7,7),
                    //               weight0_tmp(7,7),weight0_tmp(7,7),weight0_tmp(7,7),weight0_tmp(7,7),weight0_tmp(7,7),
                    //               weight0_tmp(7,7),weight0_tmp(7,7),weight0_tmp(7,7),weight0_tmp(7,7),weight0_tmp(7,7),
                    //               weight0_tmp(7,7),weight0_tmp(7,7),weight0_tmp(7,7),weight0_tmp(7,7));
                    ap_int<27> w_tmp1 = 0;
                    w_tmp1(17,0) = 0;
                    w_tmp1(25,18) = weight1_tmp;
                    w_tmp1(26,26) = weight1_tmp(7,7);

                    ap_int<27> w_tmp = w_tmp0 + w_tmp1;
                    ap_int<18> s_tmp = image_tmp;

                    ap_int<44> r_tmp = w_tmp * s_tmp;
                    IMAGE_2DT sum0_tmp = r_tmp(15,0);
                    IMAGE_2DT sum1_tmp = r_tmp(33,18) + r_tmp(17,17);
                    result_sum0 += sum0_tmp;
                    result_sum1 += sum1_tmp;
#ifdef DEBUG_CONV
                    printf("p%2d l%2d:(f0:%3d, f1:%3d)*(s:%3d) = ",p,l,(weight0_tmp . to_int()),(weight1_tmp . to_int()),(image_tmp . to_int()));
                    IMAGE_DT f0 = w_tmp(7,0);
                    IMAGE_DT f1 = w_tmp(25,18);
                    printf("(f0:%3d, f1:%3d)*(s:%3d) ",(f0 . to_int()),(f1 . to_int()),(s_tmp . to_int()));
                    printf("(s0:%3d, s1:%3d)",(sum0_tmp . to_int()),(sum1_tmp . to_int()));
                    printf("sum0:%3d, sum1:%3d ",(result_sum0 . to_int()),(result_sum1 . to_int()));
                    printf("\n");
#endif
                    #else
                    IMAGE_DT image_tmp = (buf_input(((f*PARALLEL_FILTER + l + 1) * ORG_DATA_WIDTH - 1), ((f*PARALLEL_FILTER + l) * ORG_DATA_WIDTH)));
                    IMAGE_DT w_tmp = weights_in[p][l];
                    result_sum += image_tmp * w_tmp;
#ifdef DEBUG_CONV
                    printf("l%d: (sum:%3d) += (f:%3d)*(s:%3d)",l,(result_sum . to_int()),(w_tmp . to_int()),(image_tmp . to_int()),(image_tmp . to_int()));
                    printf("\n");
#endif
                    #endif
                }
                if(init) {
                    #ifdef DSP_PACK
                    sum_buffer[p][j * FACTORS + f] = (result_sum1, result_sum0);
                    #else
                    sum_buffer[p][j * FACTORS + f] = result_sum;
                    #endif
                } else {
                    #ifdef DSP_PACK
                    IMAGE_4DT sum_buffer0 = sum_buffer[p][j * FACTORS + f](31,0);
                    IMAGE_4DT sum_buffer1 = sum_buffer[p][j * FACTORS + f](63,32);
                    sum_buffer0 += result_sum0;
                    sum_buffer1 += result_sum1;
                    sum_buffer[p][j * FACTORS + f] = (sum_buffer1, sum_buffer0);
#ifdef DEBUG_CONV
                    printf("sum_buffer0:%d, sum_buffer1:%d\n",(sum_buffer0 . to_int()),(sum_buffer1 . to_int()));
#endif
                    #else
                    sum_buffer[p][j * FACTORS + f] += result_sum;
                    #endif
                }
            }
        }
    }
}

void conv_1x1_weights(
    hls::stream< ap_int< WIDE_BUS_WIDTH > > &fifo_weights, 
#ifdef DSP_PACK
    IMAGE_2DT weights_in[PARALLEL_FILTER/2][TILING_IMAGE]
#else
    IMAGE_DT weights_in[PARALLEL_FILTER][TILING_IMAGE]
#endif
    ) 
{
    #pragma HLS inline
    #pragma ACCEL pipeline flatten
    for (int l = 0; l < TILING_IMAGE / FACTORS; l++) {
        ap_int< WIDE_BUS_WIDTH > w_buf = fifo_weights . read();
        for (int f = 0; f < FACTORS; f++) {
            ap_int< ORG_DATA_WIDTH * PARALLEL_FILTER > w_buf_sub = w_buf(((f+1)*128-1),(f*128));
            #ifdef DSP_PACK
            for (int p = 0; p < PARALLEL_FILTER / 2; p++) {
                IMAGE_2DT w_tmp = (w_buf_sub(((p + 1) * ORG_DATA_WIDTH * 2 - 1), (p * ORG_DATA_WIDTH * 2)));
            #else
            for (int p = 0; p < PARALLEL_FILTER; p++) {
                IMAGE_DT w_tmp = (w_buf_sub(((p + 1) * ORG_DATA_WIDTH - 1), (p * ORG_DATA_WIDTH)));
            #endif
                    weights_in[p][l * FACTORS + f] = w_tmp;
                    //printf("1x1_weight[%3d][%3d]=%d ", l,p,(weights_in[p][l] . to_int()));
                }
            }
            //printf("\n");
        }
}

void conv_1x1_stream(
    hls::stream< ap_int< WIDE_BUS_WIDTH > > &stream_data_in, 
    hls::stream< ap_int< WIDE_BUS_WIDTH > > &fifo_weights, 
    hls::stream< CONV_DT > stream_data_out[PARALLEL_FILTER], 
    const int config_list[32]) 
{
    int in_c = config_list[3];
    int out_w = config_list[5];
    //int out_h = config_list[6];
    int n = config_list[7];
    int one_compute_iter = config_list[26];//out_w * 13;
    int out_h_13 = config_list[27];//out_h/SPLITING_FACTOR
#ifdef DSP_PACK
    IMAGE_8DT sum_buffer[PARALLEL_FILTER/2][2*SPLITING_FACTOR*208];
    IMAGE_2DT weights_in[PARALLEL_FILTER/2][TILING_IMAGE];
#else
    IMAGE_4DT sum_buffer[PARALLEL_FILTER][2*SPLITING_FACTOR*208];
    IMAGE_DT weights_in[PARALLEL_FILTER][TILING_IMAGE];
#endif

    //one_compute_iter = out_w * 13;
    //m = out_h / 13;
    //printf("[26]one_compute_iter-0:%d,[27]out_h_13-0:%d\n", one_compute_iter, out_h_13);
    //printf("[26]one_compute_iter-1:%d,[27]out_h_13-1:%d\n", one_compute_iter, out_h_13);
    int index = 0;
    for (int h = 0; h < out_h_13; h++) {
        #pragma HLS loop_tripcount min=16 max=16
        for (int t = 0; t < n / PARALLEL_FILTER; t++) {
            #pragma HLS loop_tripcount min=2 max=2
            for (int c = 0; c < in_c / TILING_IMAGE; c++) {
                #pragma HLS loop_tripcount min=4 max=4
                printf("conv_1x1: h=%d/%d, t=%d/%d, c=%d/%d\n",h+1,out_h_13,t+1,n / 16,c+1,in_c / 16);
                conv_1x1_weights(fifo_weights, weights_in);
                conv_1x1_core(stream_data_in, weights_in, sum_buffer, one_compute_iter, c==0, c==(in_c/TILING_IMAGE-1));
                if (c == (in_c/TILING_IMAGE-1)) {
                    for (int wh = 0; wh < one_compute_iter/FACTORS; wh++) {
                        #pragma HLS loop_tripcount min=676 max=676
                        #pragma ACCEL pipeline
                        for (int f = 0; f < FACTORS; f++) {
                            if(out_w != 16 || index < 13) {
                                #ifdef DSP_PACK
                                for (int p = 0; p < PARALLEL_FILTER / 2; p++) {
                                    CONV_DT buf0_output = sum_buffer[p][wh * FACTORS + f](31,0);
                                    CONV_DT buf1_output = sum_buffer[p][wh * FACTORS + f](63,32);
                                    stream_data_out[p*2+0] . write(buf0_output);
                                    stream_data_out[p*2+1] . write(buf1_output);
#ifdef DEBUG_BURST
                                    printf("debug_1x1_result[p%d][%3d]=%5d ", p*2+0, (wh * FACTORS + f),  buf0_output . to_int());
                                    printf("debug_1x1_result[p%d][%3d]=%5d ", p*2+1, (wh * FACTORS + f),  buf1_output . to_int());
#endif
                                #else
                                for (int p = 0; p < PARALLEL_FILTER; p++) {
                                    CONV_DT buf0_output = sum_buffer[p][wh * FACTORS + f](31,0);
                                    stream_data_out[p] . write(buf0_output);
                                #endif
                                }
#ifdef DEBUG_BURST
                                printf("\n");
#endif
                            }
                            index ++;
                            if(index == out_w) 
                            index = 0;
                        }
                    }
                }
            }
        }
    }
}

void burst_in_weights(
    hls::stream< ap_int< WIDE_BUS_WIDTH > > &stream_weights, 
    int size, 
    #ifdef DSP_PACK
    IMAGE_2DT weights_in[PARALLEL_FILTER/2][3*3*TILING_IMAGE]
    #else
    IMAGE_DT weights_in[PARALLEL_FILTER][3*3*TILING_IMAGE]
    #endif
    )
{

#ifdef DEBUG_BURST
    printf("start_burst_in_weights \n");
#endif

    #pragma ACCEL pipeline flatten
    for (int i = 0; i < TILING_IMAGE * size * size / FACTORS; i++) {
        ap_int< WIDE_BUS_WIDTH > w_buf = stream_weights.read();
        for (int f = 0; f < FACTORS; f++) {
            ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > w_buf_sub = w_buf(((f+1)*ORG_DATA_WIDTH*PARALLEL_FILTER-1),(f*ORG_DATA_WIDTH*PARALLEL_FILTER));
            #ifdef DSP_PACK
            for (int p = 0; p < PARALLEL_FILTER / 2; p++){
            #else
            for (int p = 0; p < PARALLEL_FILTER; p++){
            #endif
                #ifdef DSP_PACK
                IMAGE_2DT tmp = w_buf_sub(((p + 1) * ORG_DATA_WIDTH * 2 - 1), (p * ORG_DATA_WIDTH * 2));
                #else
                IMAGE_DT tmp = w_buf_sub(((p + 1) * ORG_DATA_WIDTH - 1), (p * ORG_DATA_WIDTH));
                #endif
                weights_in[p][i * FACTORS + f] = tmp;
#ifdef DEBUG_WEIGHT
                IMAGE_DT a1 = tmp( 7,0);
                IMAGE_DT a2 = tmp(15,8);
                printf("load_weight[p%d][%3d]=%d ", p*2+0, i * FACTORS + f,(a1 . to_int()));
                printf("load_weight[p%d][%3d]=%d ", p*2+1, i * FACTORS + f,(a2 . to_int()));
#endif
            }
#ifdef DEBUG_WEIGHT
            printf("\n");
#endif
        }
    }
}

void burst_in_line(
    IMAGE_4DT line_buffer[TILING_IMAGE/4][3][420], 
    hls::stream< ap_int< WIDE_BUS_WIDTH > > &stream_in_ping, 
    hls::stream< ap_int< WIDE_BUS_WIDTH > > &stream_in_pang, 
    bool &flag_fifo, int z, int in_w, int in_h, int split_h, int size, int stride, int pad,
    int loop_w)
{
#ifdef DEBUG_BURST
    printf("start_burst_in_line \n");
    printf("in_w %d size %d pad %d, z %d, stride:%d\n",in_w,3,1,z,stride);
    printf("in_w %d in_h:%d, size %d pad %d, z %d, stride:%d\n",in_w,in_h,size,pad,z,stride);
    printf("start_burst_in_line \n");
#endif
    int loop_h = size - pad - stride + 1;
    //int loop_w = in_w / FACTORS + pad - stride + 1;
    for (int h = 0; h < loop_h; h++) {
        #pragma HLS loop_tripcount min = 2 max = 2
        #pragma ACCEL pipeline flatten
        for (int w = 0; w < loop_w; w++) {
            #pragma HLS pipeline
            #pragma HLS loop_tripcount min = 105 max = 105
            int in_h_pad = z * split_h + h - 1;
            int in_w_pad = w * FACTORS - 1;
            //printf(" in_w_pad = %d in_h_pad = %6d ", in_w_pad, in_h_pad);
            ap_int< WIDE_BUS_WIDTH > input_buf;
            bool flag_w_pad = (in_w_pad >= in_w - 1 && stride == 1);
            bool flag_h_pad = (in_h_pad < 0);
            bool flag_pad = (flag_w_pad || flag_h_pad);

            if (flag_pad == 0){
                if (w == 0) {
                    flag_fifo = !flag_fifo;
                }
                if (flag_fifo) {
                    input_buf = stream_in_ping . read();
                }
                else {
                    input_buf = stream_in_pang . read();
                }
            } else{
                input_buf = 0;
            }
            IMAGE_4DT tmp_value[TILING_IMAGE/4];

            if(flag_w_pad == 0){
                for(int f = 0; f < FACTORS; f++){
                    merlinL41:
                    for (int l = 0; l < TILING_IMAGE/4; l++) {
                        tmp_value[l] = input_buf(((f*TILING_IMAGE + (l + 1)*4) * ORG_DATA_WIDTH - 1), ((f*TILING_IMAGE + l*4) * ORG_DATA_WIDTH));
                        line_buffer[l][h + stride][f + w*FACTORS] = tmp_value[l];
                    }
#ifdef DEBUG_BURST
                    ap_int< 8 > a = (line_buffer[0][h + stride][w*4+f](7, 0));
                    printf("z%d burst_in [%3d]line_buffer[%3d][%3d] = %3d ", z,flag_fifo,h,w*4+f,(a . to_int()));
#endif
                }
            } else {
                for (int l = 0; l < TILING_IMAGE/4; l++) {
                    line_buffer[l][h + stride][0 + w*FACTORS] = 0;
                }
#ifdef DEBUG_BURST
                ap_int< 8 > a = (line_buffer[0][h + stride][w * 4 + 0](7,0));
                printf("z%d burst_in [%3d]line_buffer[%3d][%3d] = %3d ",z,(int )((bool )flag_fifo),h,w * 4 + 0,(a . to_int()));
#endif
            }
        }
#ifdef DEBUG_BURST
        printf("\n");
#endif
    }
}

void shift_in_data(
    hls::stream< ap_int< WIDE_BUS_WIDTH > > &stream_in_ping, 
    hls::stream< ap_int< WIDE_BUS_WIDTH > > &stream_in_pang, 
    hls::stream< ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > > stream_shift_in[2][4], 
    bool &flag_fifo, int z, int in_w, int in_h, int split_h, int size, int stride, int pad, int new_h,
    int loop_h, int loop_w)
{
#ifdef DEBUG_BURST
    printf("start_shift_in_data \n");
    printf("in_w %d size %d pad %d, z %d, stride:%d\n",in_w,3,1,z,stride);
    printf("in_w %d in_h:%d, size %d pad %d, z %d, stride:%d\n",in_w,in_h,3,1,z,stride);
#endif
    //int loop_h = new_h / stride;
    //int loop_w = in_w / FACTORS + pad - stride + 1;
    for (int h = 0; h < loop_h; h++) {
        #pragma HLS loop_tripcount min = 13 max = 13
        #pragma ACCEL pipeline flatten
        for (int w = 0; w < loop_w; w++) {
            #pragma HLS loop_tripcount min = 105 max = 105
            int in_w_pad = w * FACTORS;
            int in_h_pad = z * split_h + h - 1;
            //printf("in_w_pad:%3d, in_h_pad:%3d new_h:%3d ", in_w_pad, in_h_pad, new_h);
            ap_int< WIDE_BUS_WIDTH > input_buf[2];
            bool flag_w_pad = (in_w_pad >= in_w && stride == 1);
            bool flag_h_pad = (in_h_pad >= in_h - 3 + 1 - stride + 1);
            bool flag_pad = (flag_w_pad || flag_h_pad);

            if (flag_pad == 0) {
                if (stride == 2) {
                    if (flag_fifo) {
                        input_buf[0] = stream_in_pang . read();
                        input_buf[1] = stream_in_ping . read();
                    }
                    else {
                        input_buf[0] = stream_in_ping . read();
                        input_buf[1] = stream_in_pang . read();
                    }
                } else {
                    if (w == 0) {
                        flag_fifo = !flag_fifo;
                    }
                    if (flag_fifo) {
                        input_buf[0] = stream_in_ping . read();
                        input_buf[1] = 0;
                    }
                    else {
                        input_buf[0] = stream_in_pang . read();
                        input_buf[1] = 0;
                    }
                }
            } else {
                input_buf[0] = 0;
                input_buf[1] = 0;
            }
            if(flag_w_pad == 0){
                ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > buf_in[2][4];

                buf_in[0][0] = input_buf[0]((128 * 0 + 127),(128 * 0));
                buf_in[0][1] = input_buf[0]((128 * 1 + 127),(128 * 1));
                buf_in[0][2] = input_buf[0]((128 * 2 + 127),(128 * 2));
                buf_in[0][3] = input_buf[0]((128 * 3 + 127),(128 * 3));
                buf_in[1][0] = input_buf[1]((128 * 0 + 127),(128 * 0));
                buf_in[1][1] = input_buf[1]((128 * 1 + 127),(128 * 1));
                buf_in[1][2] = input_buf[1]((128 * 2 + 127),(128 * 2));
                buf_in[1][3] = input_buf[1]((128 * 3 + 127),(128 * 3));
                if(in_w != 16 || w < 4) {
                    stream_shift_in[0][0] . write(buf_in[0][0]);
                    stream_shift_in[0][1] . write(buf_in[0][1]);
                }
                if(in_w != 16 || w < 3) {
                    stream_shift_in[0][2] . write(buf_in[0][2]);
                    stream_shift_in[0][3] . write(buf_in[0][3]);
                }
                if(in_w != 16 || w < 4) {
                    stream_shift_in[1][0] . write(buf_in[1][0]);
                    stream_shift_in[1][1] . write(buf_in[1][1]);
                }
                if(in_w != 16 || w < 3) {
                    stream_shift_in[1][2] . write(buf_in[1][2]);
                    stream_shift_in[1][3] . write(buf_in[1][3]);
                }
            } else {
                if(in_w != 16) {
                    stream_shift_in[0][0] . write(0);
                    stream_shift_in[1][0] . write(0);
                }
            }
        }
    }
#ifdef DEBUG_BURST
    printf("finish_shift_in_data\n");
#endif
}

void conv_3x3_core(
    IMAGE_4DT shift[TILING_IMAGE/4][3][3],
    #ifdef DSP_PACK
    IMAGE_2DT weights_in[TILING_IMAGE*3*3], 
    hls::stream< IMAGE_4DT > stream_sum_out[2], 
    #else
    IMAGE_DT weights_in[TILING_IMAGE*3*3], 
    hls::stream< IMAGE_4DT > &stream_sum_out, 
    #endif
    int size)
{
    #ifdef DSP_PACK
    IMAGE_4DT result_sum0 = 0;
    IMAGE_4DT result_sum1 = 0;
    #else
    IMAGE_4DT result_sum = 0;
    #endif
    for (int l = 0; l < TILING_IMAGE / 4; l++) {
        for (int k = 0; k < 4; k++) {
            for (int w = 0; w < size; w++) {
                for (int h = 0; h < size; h++) {
                #ifdef DSP_PACK
                    IMAGE_DT image_tmp = shift[l][size - w - 1][h]((k+1)*8-1, k*8);
                    IMAGE_2DT weights_tmp = weights_in[(l * 4 + k) * size * size + h * size + w];
                    IMAGE_DT weight0_tmp = weights_tmp(7,0);
                    IMAGE_DT weight1_tmp = weights_tmp(15,8);
                    ap_int<27> w_tmp0 = 0;
                    w_tmp0(7,0)= weight0_tmp;
                    w_tmp0(26,8) = (weight0_tmp(7,7) == 1) ? 0x7ffff : 0;
                    //w_tmp0(26,8) = (weight0_tmp(7,7),weight0_tmp(7,7),weight0_tmp(7,7),weight0_tmp(7,7),weight0_tmp(7,7),
                    //               weight0_tmp(7,7),weight0_tmp(7,7),weight0_tmp(7,7),weight0_tmp(7,7),weight0_tmp(7,7),
                    //               weight0_tmp(7,7),weight0_tmp(7,7),weight0_tmp(7,7),weight0_tmp(7,7),weight0_tmp(7,7),
                    //               weight0_tmp(7,7),weight0_tmp(7,7),weight0_tmp(7,7),weight0_tmp(7,7));
                    ap_int<27> w_tmp1 = 0;
                    w_tmp1(25,18) = weight1_tmp;
                    w_tmp1(26,26) = weight1_tmp(7,7);

                    ap_int<27> w_tmp = w_tmp0 + w_tmp1;
                    ap_int<18> s_tmp = image_tmp;

                    ap_int<44> r_tmp = w_tmp * s_tmp;
                    IMAGE_2DT sum0_tmp = r_tmp(15,0);
                    IMAGE_2DT sum1_tmp = r_tmp(33,18) + r_tmp(17,17);
                    result_sum0 += sum0_tmp;
                    result_sum1 += sum1_tmp;
#ifdef DEBUG_CONV
                    printf("l%d:(f0:%3d, f1:%3d)*(s:%3d) = ",l,(weight0_tmp . to_int()),(weight1_tmp . to_int()),(shift_tmp . to_int()));
                    IMAGE_DT f0 = w_tmp(7,0);
                    IMAGE_DT f1 = w_tmp(25,18);
                    printf("(f0:%3d, f1:%3d)*(s:%3d) ",(f0 . to_int()),(f1 . to_int()),(s_tmp . to_int()));
                    printf("s0:%d, s1:%d ",(sum0_tmp . to_int()),(sum1_tmp . to_int()));
                    printf("sum0:%d, sum1:%d ",(result_sum0 . to_int()),(result_sum1 . to_int()));
                    printf("\n");
#endif
                #else
                    IMAGE_DT shift_tmp = shift[l][size - w - 1][h]((k+1)*8-1, k*8);
                    IMAGE_DT weights_in_tmp = weights_in[(l * 4 + k) * size * size + h * size + w];
                    result_sum += shift_tmp * weights_in_tmp;
                #endif
                }
#ifdef DEBUG_CONV
                printf("\n");
#endif
            }
        }
#ifdef DEBUG_CONV
        printf("\n");
#endif
    }
    #ifdef DSP_PACK
    stream_sum_out[0] . write(result_sum0);
    stream_sum_out[1] . write(result_sum1);
    #else
    stream_sum_out . write(result_sum);
    #endif
}    

void shift_reg(
    IMAGE_4DT shift[TILING_IMAGE/4][3][3],
    IMAGE_4DT line_buffer[TILING_IMAGE/4][3][420], 
    hls::stream< ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > > stream_shift_in[2][4],
    int stride,
    int size,
    int i) 
{
    if (i == 0) {
        for (int l = 0; l < TILING_IMAGE / 4; l++){ 
            for (int w = 0; w < size; w++) {
                for (int h = 0; h < size; h++){ 
                    shift[l][w][h] = 0;
                }
            }
        }
    }

    if (stride == 2) {
        ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > input_buf[2][2];
        input_buf[0][0] = stream_shift_in[0][2*(i%2)+0] . read();
        input_buf[0][1] = stream_shift_in[0][2*(i%2)+1] . read();
        input_buf[1][0] = stream_shift_in[1][2*(i%2)+0] . read();
        input_buf[1][1] = stream_shift_in[1][2*(i%2)+1] . read();
        for (int l = 0; l < TILING_IMAGE/4; l++) {
            for (int st = 0; st < 2; st++) {
                line_buffer[l][0][i * 2 + st] = line_buffer[l][2][i * 2 + st];
            }
#ifdef DEBUG_CONV
            if(l == 0){
                for (int h = 0; h < size; h++) {
                    for (int w = 0; w < 418; w++) {
                        IMAGE_DT a = line_buffer[0][h][w];
                        printf("input1[0][%d][%d] = %d ", h, w,(a . to_int()));
                    }
                    printf("\n");
                }
                printf("\n");
            }
#endif
            line_buffer[l][1][i * 2 + 0] = input_buf[0][0]((((l + 1)*4) * ORG_DATA_WIDTH - 1), ((l*4) * ORG_DATA_WIDTH));
            line_buffer[l][1][i * 2 + 1] = input_buf[0][1]((((l + 1)*4) * ORG_DATA_WIDTH - 1), ((l*4) * ORG_DATA_WIDTH));
            line_buffer[l][2][i * 2 + 0] = input_buf[1][0]((((l + 1)*4) * ORG_DATA_WIDTH - 1), ((l*4) * ORG_DATA_WIDTH));
            line_buffer[l][2][i * 2 + 1] = input_buf[1][1]((((l + 1)*4) * ORG_DATA_WIDTH - 1), ((l*4) * ORG_DATA_WIDTH));
#ifdef DEBUG_CONV
            if(l == 0){
                for (int h = 0; h < size; h++) {
                    for (int w = 0; w < 418; w++) {
                        IMAGE_DT a = line_buffer[0][h][w];
                        printf("input2[0][%d][%d] = %d ", h, w,(a . to_int()));
                    }
                    printf("\n");
                }
                printf("\n");
            }
#endif

            for (int h = 0; h < size; h++) {
                shift[l][2][h] = shift[l][0][h];
            }
            for (int st = 0; st < 2; st++) {
                for (int h = 0; h < size; h++) {
                    shift[l][1 - st][h] = line_buffer[l][h][i * 2 + st];
                }
            }
        }
    } else {
        ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > input_buf[2];
        input_buf[0] = stream_shift_in[0][i%4] . read();
        input_buf[1] = stream_shift_in[1][i%4] . read();
        for (int l = 0; l < TILING_IMAGE/4; l++) {
            for (int h = 0; h < size - 1; h++) {
                line_buffer[l][h][i] = line_buffer[l][h + 1][i];
            }
#ifdef DEBUG_CONV
            if(l == 0){
                for (int h = 0; h < size; h++) {
                    for (int w = 0; w < 418; w++) {
                        IMAGE_DT a = line_buffer[0][h][w];
                        printf("input0[0][%d][%d] = %d ", h, w,(a . to_int()));
                    }
                    printf("\n");
                }
                printf("\n");
            }
#endif
            ap_int< ORG_DATA_WIDTH*4 > tmp_buf = input_buf[0]((((l + 1)*4) * ORG_DATA_WIDTH - 1), (((l*4) * ORG_DATA_WIDTH)));
            line_buffer[l][size - 1][i] = tmp_buf;
#ifdef DEBUG_CONV
            if(l == 0){
                for (int h = 0; h < size; h++) {
                    for (int w = 0; w < 418; w++) {
                        IMAGE_DT a = line_buffer[0][h][w];
                        printf("input1[0][%d][%d] = %d ", h, w,(a . to_int()));
                    }
                    printf("\n");
                }
                printf("\n");
            }
#endif
            for (int w = 0; w < size - 1; w++) {
                int k = size - 1 - w;
                for (int h = 0; h < size; h++) {
                    shift[l][k][h] = shift[l][k - 1][h];
                }
            }
            for (int h = 0; h < size; h++) {
                shift[l][0][h] = line_buffer[l][h][i];
            }
        }
    }
#ifdef DEBUG_CONV
    for (int h = 0; h < size; h++) {
       for (int w = 0; w < size; w++) {
           IMAGE_DT a = shift[0][size - 1 - w][h];
           printf("shift_ok[%d] = %d ", h * size + w,(a . to_int()));
       }
       printf("\n");
    }
#endif
}

void compute_one_cube(
    IMAGE_4DT line_buffer[TILING_IMAGE/4][3][420], 
    #ifdef DSP_PACK
    IMAGE_2DT weights_in[PARALLEL_FILTER/2][TILING_IMAGE*3*3], 
    #else
    IMAGE_DT weights_in[PARALLEL_FILTER][TILING_IMAGE*3*3], 
    #endif
    hls::stream< ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > > stream_shift_in[2][4], 
    #ifdef DSP_PACK
    hls::stream< IMAGE_4DT > stream_sum_out[PARALLEL_FILTER/2][2], 
    #else
    hls::stream< IMAGE_4DT > stream_sum_out[PARALLEL_FILTER], 
    #endif
    int z, int in_w, int in_h, int size, int stride, int pad, int new_h,
    int loop_h, int loop_w)
{
#ifdef DEBUG_CONV
    printf("start_compute_one_plan_sub \n");
    printf("in_w %d in_h %d, size %d pad %d, z %d, stride:%d\n",in_w, in_h,3,1,z,stride);
    printf("z:%d, new_h:%d\n",z, new_h);
    printf("h loop:%d, w loop:%d\n", new_h / stride,((in_w + pad - stride + 1) / stride + FACTORS / stride - 1) / (FACTORS / stride));
#endif
    
    IMAGE_4DT shift[TILING_IMAGE/4][3][3];
    //int loop_h = new_h / stride;
    //int loop_w = (in_w + pad - stride + 1) / stride;
    if(loop_w == 17) {
        loop_w = 14;
        in_w = 13;
    }
    for (int h = 0; h < loop_h; h++) {
        #pragma HLS loop_tripcount min = 13 max = 13
        #pragma ACCEL pipeline flatten
        for (int w = 0; w < loop_w; w++) {
            #pragma HLS loop_tripcount min = 420 max = 420
            //printf("h:%d/%d, w:%d/%d\n",h+1,new_h / stride,w+1,((in_w + 1 - stride + 1) / stride + WIDE_BUS_WIDTH / 8 / 16 / stride - 1) / (512 / 8 / 16 / stride));
            shift_reg(shift, line_buffer, stream_shift_in, stride, size, w);
            
#ifdef DEBUG_CONV
            printf("count:%3d %3d %3d\n", h, w, (in_w + pad - stride + 1)/stride);
#endif
            //if (w >= size - pad - stride && w < (in_w + pad - stride + 1)/stride) {
            if (w >= size - pad - stride) {
                #pragma ACCEL parallel
                #ifdef DSP_PACK
                for(int p = 0; p < PARALLEL_FILTER / 2; p++) {
                #else
                for(int p = 0; p < PARALLEL_FILTER; p++) {
                #endif
#ifdef DEBUG_CONV
                    printf("p:%3d\n", p);
#endif
                    conv_3x3_core(shift, weights_in[p], stream_sum_out[p], size);
                }
            }
        }
    }
#ifdef DEBUG_CONV
    printf("finish_compute_one_plan_sub\n");
#endif
}

void adder_out(
    #ifdef DSP_PACK
    hls::stream< IMAGE_4DT > stream_data_in[PARALLEL_FILTER/2][2], 
    #else
    hls::stream< IMAGE_4DT > stream_data_in[PARALLEL_FILTER], 
    #endif
    hls::stream< CONV_DT > stream_data_out[PARALLEL_FILTER], 
    #ifdef RUNSIM
    IMAGE_4DT conv_sum[PARALLEL_FILTER][SPLITING_FACTOR*416],
    #endif
    int z, int s, int in_w, int in_c, int size, int stride, int pad, int new_h,
    int h_col, int w_col)
{

    #ifdef BITGEN
    IMAGE_4DT conv_sum[PARALLEL_FILTER][SPLITING_FACTOR*416];
    #endif
    //int h_col = (new_h + 2 * pad - size) / stride + 1;
    //int w_col = (in_w + 2 * pad - size) / stride + 1;
    if(w_col == 16) { w_col = 13; }
#ifdef DEBUG_BURST
    printf("start_adder_out \n");
    printf("z:%d, s:%d, in_w %d size %d pad %d, stride:%d\n",z,s,in_w,size,pad,stride);
    printf("w_col:%d, h_col:%d\n",w_col,h_col);
#endif
    for (int i = 0; i < h_col * w_col; i++) {
        #pragma HLS pipeline
        #pragma HLS loop_tripcount min = 5408 max = 5408
        #pragma ACCEL parallel
        #ifdef DSP_PACK
        for (int p = 0; p < PARALLEL_FILTER / 2; p++) {
            IMAGE_4DT tmp0 = stream_data_in[p][0] . read();
            IMAGE_4DT tmp1 = stream_data_in[p][1] . read();
        #else
        for (int p = 0; p < PARALLEL_FILTER; p++) {
            IMAGE_4DT tmp0 = stream_data_in[p] . read();
        #endif
            if (s == 0) {
                #ifdef DSP_PACK
                conv_sum[p*2+0][i] = tmp0;
                conv_sum[p*2+1][i] = tmp1;
                #else
                conv_sum[p][i] = tmp0;
                #endif
            }
            else {
                #ifdef DSP_PACK
                conv_sum[p*2+0][i] += tmp0;
                conv_sum[p*2+1][i] += tmp1;
                #else
                conv_sum[p][i] += tmp0;
                #endif
            }
            if (s == in_c / TILING_IMAGE - 1) {
                #ifdef DSP_PACK
                CONV_DT buf0_output = conv_sum[p*2+0][i];
                CONV_DT buf1_output = conv_sum[p*2+1][i];
                stream_data_out[p*2+0] . write(buf0_output);
                stream_data_out[p*2+1] . write(buf1_output);
                #else
                CONV_DT buf0_output = conv_sum[p][i];
                stream_data_out[p] . write(buf0_output);
                #endif
#ifdef DEBUG_CONV
                #ifdef DSP_PACK
                printf("conv_sum[%3d][%3d] = (%3d, %3d) ", i / w_col, i % w_col, (buf0_output . to_int()), (buf1_output . to_int()));
                #else
                printf("conv_sum[%3d][%3d] = %3d ", i / w_col, i % w_col, (buf0_output . to_int()));
                #endif
                }
#endif
            }
#ifdef DEBUG_CONV
            else {
                CONV_DT buf0_output = conv_sum[p*2+0][i];
                CONV_DT buf1_output = conv_sum[p*2+1][i];
                #ifdef DSP_PACK
                printf("[p%d] conv_sum_tmp[%3d][%3d] = (%3d, %3d) ", p,i / w_col, i % w_col, (buf0_output . to_int()), (buf1_output . to_int()));
                #else
                printf("[p%d] conv_sum_tmp[%3d][%3d] = %3d ", p, i / w_col, i % w_col, (buf0_output . to_int()));
                #endif
                }
            }
#endif
        }
#ifdef DEBUG_CONV
    printf("\n");
#endif
    }
#ifdef DEBUG_CONV
    printf("finish_adder_out \n");
#endif
}

void conv_3x3_cuboid(
    IMAGE_4DT line_buffer[TILING_IMAGE/4][3][420],
    hls::stream< ap_int< WIDE_BUS_WIDTH > > &stream_in_ping, 
    hls::stream< ap_int< WIDE_BUS_WIDTH > > &stream_in_pang, 
    #ifdef DSP_PACK
    IMAGE_2DT weights_in[PARALLEL_FILTER/2][TILING_IMAGE*3*3], 
    #else
    IMAGE_DT weights_in[PARALLEL_FILTER][TILING_IMAGE*3*3], 
    #endif
    hls::stream< CONV_DT > stream_out[PARALLEL_FILTER], 
    bool &flag_fifo, int z, int s, int in_w, int in_h, int in_c, int split_h, int size, int stride, int pad,
    int loop_1, int loop_2, int loop_3, int h_col, int w_col,
    int cond, int new_h_1, int new_h_2)
{
    int new_h;
    if (z == cond) { 
        new_h = new_h_2; 
    } else {
        new_h = new_h_1; 
    }
    if (stride == 2) {
        loop_1 = new_h >> 1;
        h_col = new_h >> 1;
        //h_col = ((new_h - 1) >> 1) + 1;
    }
#ifdef DEBUG_BURST
    printf("conv 3x3: split_h=%d\n", split_h);
    printf("loop_1=%d, loop_2=%d, loop_3=%d\n", loop_1, loop_2, loop_3);
    printf("cond=%d, new_h_1=%d, new_h_2=%d\n", cond, new_h_1, new_h_2);
    printf("w_col=%d, h_col=%d\n",w_col,h_col);
#endif
    hls::stream< ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > > stream_shift_in[2][4];
    #pragma HLS stream variable = stream_shift_in depth = 512
    #ifdef DSP_PACK
    hls::stream< IMAGE_4DT > stream_sum_out[PARALLEL_FILTER/2][2];
    #else
    hls::stream< IMAGE_4DT > stream_sum_out[PARALLEL_FILTER];
    #endif
    #pragma HLS stream variable = stream_sum_out depth = 512
    #ifdef RUNSIM
    IMAGE_4DT conv_sum[PARALLEL_FILTER][SPLITING_FACTOR*416];
    #endif
    #pragma HLS dataflow
    shift_in_data(stream_in_ping, stream_in_pang, stream_shift_in, flag_fifo, z, in_w, in_h, split_h, size, stride, pad, new_h, loop_1, loop_2);
    compute_one_cube(line_buffer, weights_in, stream_shift_in, stream_sum_out, z, in_w, in_h, size, stride, pad, new_h, loop_1, loop_3);
    #ifdef RUNSIM
    adder_out(stream_sum_out, stream_out, conv_sum, z, s, in_w, in_c, size, stride, pad, new_h,h_col, w_col);
    #else
    adder_out(stream_sum_out, stream_out, z, s, in_w, in_c, size, stride, pad, new_h,h_col, w_col);
    #endif
}

void conv_3x3_stream(
    hls::stream< ap_int< WIDE_BUS_WIDTH > > &stream_in_ping,                       
    hls::stream< ap_int< WIDE_BUS_WIDTH > > &stream_in_pang,
    hls::stream< ap_int< WIDE_BUS_WIDTH > > &stream_weights, 
    hls::stream< CONV_DT > stream_out[PARALLEL_FILTER], 
    const int config_list[32]) 
{
    int size = 3;//config_list[0];
    int in_w = config_list[1];
    int in_h = config_list[2];
    int in_c = config_list[3];
    int n = config_list[7];
    int stride = config_list[9];
    int pad = 1;//config_list[10];
    int split_h = config_list[20];
    int new_h_2 = config_list[23];
    int in_h_13 = config_list[27];
    bool flag_fifo = 0;
#ifdef DEBUG_BURST
    printf("in_w:%d, in_h:%d, in_c:%d n:%d stride:%d, in_h_13:%d, split_h:%d\n",in_w,in_h,in_c,n,stride,in_h_13, split_h);
#endif
    int new_h_1 = split_h - size + 1 + 2 * pad + stride - 1;
    //int new_h_2 = (in_h + 2 * pad - size) % (split_h + stride - 1) + 1;
    //int cond = in_h / (split_h + stride - 1);
    int cond = config_list[21];
    int loop_1 = new_h_1;
    int loop_2 = in_w / FACTORS + 1 - stride + 1;
    int loop_3 = config_list[12];
    //int loop_3 = (in_w + 1 - stride + 1) / stride;
    //int h_col = (new_h_1 + 2 * 1 - 3) / stride + 1;
    int h_col = new_h_1;
    int w_col = config_list[25];
    //int w_col = (in_w + 2 * 1 - 3) / stride + 1;

    //caculate PARALLEL_FILTER filter core once
    for (int t = 0; t < n / PARALLEL_FILTER; t++) {
        #pragma HLS loop_tripcount min = 2 max = 2
        for (int z = 0; z < in_h_13; z++) {
            #pragma HLS loop_tripcount min = 32 max = 32
            for (int s = 0; s < in_c / TILING_IMAGE; s++) {
                printf("conv 3x3: t %d/%d, z loop %d/%d, s %d/%d\n", t+1, n/PARALLEL_FILTER, z+1, in_h_13, s+1, in_c/TILING_IMAGE);
                #pragma HLS loop_tripcount min = 2 max = 2
                #ifdef DSP_PACK
                IMAGE_2DT weights_in[PARALLEL_FILTER/2][TILING_IMAGE*3*3];
                #else
                IMAGE_DT weights_in[PARALLEL_FILTER][TILING_IMAGE*3*3];
                #endif
                IMAGE_4DT line_buffer[PARALLEL_FILTER/4][3][420];

                burst_in_weights(stream_weights, size, weights_in);
                burst_in_line(line_buffer, stream_in_ping, stream_in_pang, flag_fifo, z, in_w, in_h, split_h, size, stride, pad, loop_2);
                conv_3x3_cuboid(line_buffer, stream_in_ping, stream_in_pang, weights_in, stream_out, flag_fifo, z, s, in_w, in_h, in_c, split_h, size, stride, pad, loop_1, loop_2, loop_3, h_col, w_col,cond, new_h_1,new_h_2);
            }
        }
    }
}

void conv_switch(
    hls::stream< ap_int< WIDE_BUS_WIDTH > > &stream_in_ping,
    hls::stream< ap_int< WIDE_BUS_WIDTH > > &stream_in_pang,
    hls::stream< ap_int< WIDE_BUS_WIDTH > > &stream_in_1x1,
    hls::stream< ap_int< WIDE_BUS_WIDTH > > &fifo_weights_3x3,
    hls::stream< ap_int< WIDE_BUS_WIDTH > > &fifo_weights_1x1,
    hls::stream< CONV_DT > stream_out[PARALLEL_FILTER],
    const int config_list[32]) 
{
    #pragma HLS inline off
    int conv_en = config_list[0];
    if (conv_en == 3) {
        conv_3x3_stream(stream_in_ping, stream_in_pang, fifo_weights_3x3, stream_out, config_list);
    } else {
        conv_1x1_stream(stream_in_1x1, fifo_weights_1x1, stream_out, config_list);
    }
}

void bias_stream(
    hls::stream< CONV_DT > stream_input[PARALLEL_FILTER],
    hls::stream< ap_int< WIDE_BUS_WIDTH > > & fifo_bias, 
    hls::stream< ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > > & stream_output, 
    const int config_list[32])
{
    //int batch_normalize = config_list[12];
    int in_w = config_list[5];
    int n = config_list[7];
    int right_shift_cnt = config_list[11];
    int activation = config_list[14];
    int in_wh = config_list[18];
#ifdef DEBUG_BIAS
    int batch_normalize = config_list[12];
    printf("in_wh:%d ", in_wh);
    printf("n:%d, batch_normalize:%d, activation:%d\n",n,batch_normalize,activation);
#endif
    IMAGE_4DT biases[PARALLEL_FILTER];
    for (int t = 0; t < n / PARALLEL_FILTER; t++) {
        #pragma HLS loop_tripcount min = 2 max = 2
        ap_int< WIDE_BUS_WIDTH > bias_buf = fifo_bias . read();
        for (int p = 0; p < PARALLEL_FILTER; p++) {
            biases[p] = bias_buf(((p + 1) * 32 - 1), (p * 32));

#ifdef DEBUG_BIAS
            printf("biases[%d]:%d\n", p, biases[p].to_int());
#endif
        }
        int index = 0;
        for (int i = 0; i < in_wh; i++) {
            #pragma HLS pipeline
            #pragma HLS loop_tripcount min = 173056 max =173056
            ap_int< ORG_DATA_WIDTH > tmp_a[PARALLEL_FILTER];
            if(index > 12 && in_w == 16) {
                stream_output . write(0);
            } else {
                for (int p = 0; p < PARALLEL_FILTER; p++) {
                    CONV_DT input_buf = stream_input[p] . read();
                    CONV_DT sum0 = input_buf + biases[p];
                    IMAGE_DT sum1 = xilinx_quantizer_shift(sum0, right_shift_cnt);
                    IMAGE_DT sum2;
                    if (activation == 1) {
                        if (sum1 < 0) {
                            IMAGE_4DT sum3 = sum1 * 104;
                            sum2 = xilinx_quantizer_shift(sum3, 10);
                        } else {
                            IMAGE_4DT sum3 = sum1;
                            sum2 = xilinx_quantizer_shift(sum3, 0);
                        }
                    } else {
                        sum2 = sum1;
                    }
                    tmp_a[p] = sum2;
#ifdef DEBUG_BIAS
                    printf("[p%2d]debug_bias_data[%d] : bias[%3d] + in[%3d] = [%3d] -> q[%3d] -> act[%3d] ",p,i, biases[p].to_int(), input_buf.to_int(), sum0.to_int(), sum1.to_int(), sum2.to_int());
#endif
                }
#ifdef DEBUG_BIAS
                printf("\n");
#endif
                ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > output_buf = ((((((((((((((((tmp_a[15] , tmp_a[14] , tmp_a[13])) , tmp_a[12] , tmp_a[11])) , tmp_a[10] , tmp_a[9])) , tmp_a[8] , tmp_a[7])) , tmp_a[6] , tmp_a[5])) , tmp_a[4] , tmp_a[3])) , tmp_a[2] , tmp_a[1])) , tmp_a[0]));
                stream_output . write(output_buf);
            }
            index++;
            if(index == in_w) {
                index = 0;
            }
        }
    }
}

void bias_switch(
    hls::stream< CONV_DT > stream_input[PARALLEL_FILTER],
    hls::stream< ap_int< WIDE_BUS_WIDTH > > & fifo_bias, 
    hls::stream< ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > > & stream_output, 
    const int config_list[32]) 
{

    #pragma HLS inline off
    #ifdef DEBUG_BIAS
    printf("start bias \n");
    #endif
    //int conv_en = config_list[0];
    //if (conv_en == 1 || conv_en == 3) {
    bias_stream(stream_input, fifo_bias,stream_output,config_list);
    //}
    #ifdef DEBUG_BIAS
    printf("finish_bias\n");
    #endif
}

void shortcut_core(
    ap_int< WIDE_BUS_WIDTH > *buf_in_image, 
    hls::stream< ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > > &stream_input, 
    hls::stream< ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > > &stream_output, 
    int burst_length, int in_w,
    int right_shift_cnt0, int right_shift_cnt1, int right_shift_cnt2)
{
    merlinL77:
    for (int i = 0; i < burst_length/FACTORS; i++) {
        #pragma HLS loop_tripcount min = 832 max = 832
        for (int f = 0; f < FACTORS; f++){
            #pragma HLS pipeline
            ap_int< ORG_DATA_WIDTH > tmp_a[PARALLEL_FILTER];
            ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > input_buf1 = stream_input . read();
            ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > input_buf2 = (buf_in_image[i]((f+1)*ORG_DATA_WIDTH*PARALLEL_FILTER-1, f*ORG_DATA_WIDTH*PARALLEL_FILTER));
            merlinL76:
            for (int p = 0; p < PARALLEL_FILTER; p++) {
                ap_int< ORG_DATA_WIDTH > tmp_buf1 = (input_buf1(((p + 1) * ORG_DATA_WIDTH - 1), (p * ORG_DATA_WIDTH)));
                ap_int< ORG_DATA_WIDTH > tmp_buf2 = (input_buf2(((p + 1) * ORG_DATA_WIDTH - 1), (p * ORG_DATA_WIDTH)));
                IMAGE_4DT tmp_1 = tmp_buf1;
                IMAGE_4DT tmp_2 = tmp_buf2;
                IMAGE_4DT tmp_o1 = tmp_1 << right_shift_cnt0;
                IMAGE_4DT tmp_o2 = tmp_2 << right_shift_cnt1;
                IMAGE_4DT tmp_o = (tmp_o1 + tmp_o2);
                IMAGE_DT sum1 = xilinx_quantizer_shift(tmp_o, right_shift_cnt2);
                tmp_a[p] = sum1;
#ifdef DEBUG_SHORTCUT
                printf("[p%2d]shortcut_data[%4d]:(%3d->%3d) + (%3d->%3d) ", p, i*4+f,(tmp_buf1 . to_int()),(tmp_o1 . to_int()),(tmp_buf2 . to_int()),(tmp_o2 . to_int()));
                printf(" =  (%3d->%3d) ", (tmp_o . to_int()),(sum1 . to_int()));
#endif
            }
#ifdef DEBUG_SHORTCUT
                printf("\n");
#endif

            ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > output_buf = ((((((((((((((((tmp_a[15] , tmp_a[14] , tmp_a[13])) , tmp_a[12] , tmp_a[11])) , tmp_a[10] , tmp_a[9])) , tmp_a[8] , tmp_a[7])) , tmp_a[6] , tmp_a[5])) , tmp_a[4] , tmp_a[3])) , tmp_a[2] , tmp_a[1])) , tmp_a[0]));
            stream_output . write(output_buf);
        }
    }

}
void shortcut_stream(
    ap_int< WIDE_BUS_WIDTH > *in_image,
    hls::stream< ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > > &stream_input, 
    hls::stream< ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > > &stream_output, 
    const int config_list[32])
{
   //printf("start_shortcut_stream\n");
    int in_w = config_list[5];
    int right_shift_cnt0 = config_list[11];
    int right_shift_cnt1 = config_list[12];
    int right_shift_cnt2 = config_list[13];
    int burst_length = config_list[24];
    int loop_bound = config_list[25];
    int input_idx = config_list[28];
    //printf("shortcut_stream-0:burst_length = %d\n", burst_length);
    //printf("shortcut_stream-0:loop_bound = %d\n", loop_bound);
    //printf("shortcut:input_idx:%d\n", input_idx);
    //int burst_length = 13*208;
    //if(in_w == 28)
    //    burst_length = 13 * in_w * 8;
    //else if(in_w == 16|| in_w == 14)
    //    burst_length = 13 * in_w * 16;
    //data_size = config_list[5] * config_list[6] * config_list[7];
    //loop_bound = data_size / PARALLEL_FILTER /burst_length;
    //printf("shortcut_stream-1:burst_length = %d\n", burst_length);
    //printf("shortcut_stream-1:loop_bound = %d\n", loop_bound);
    ap_int< WIDE_BUS_WIDTH > buf_in_image[13*16*16/4];
    merlinL78:
    for (int t = 0; t < loop_bound; t++) {
        #pragma HLS loop_tripcount min = 64 max = 64
        memcpy((void*)buf_in_image, (void*)&in_image[t*burst_length/4 + input_idx], sizeof(ap_int< WIDE_BUS_WIDTH >) * burst_length/4);
        shortcut_core(buf_in_image, stream_input, stream_output, burst_length, in_w, right_shift_cnt0, right_shift_cnt1, right_shift_cnt2);
    }
}

void shortcut_switch(
    ap_int< WIDE_BUS_WIDTH > *in_image,
    hls::stream< ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > > &stream_input, 
    hls::stream< ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > > &stream_output, 
    const int config_list[32])
{
    #pragma HLS inline off
    int shortcut_sel = config_list[0];
    if (shortcut_sel == 1) {
        shortcut_stream(in_image, stream_input,stream_output,config_list);
    }
    else {
        int data_size = config_list[19];
        //int data_size = config_list[5] * config_list[6] * config_list[7];
        //printf("max pool data size %d %d\n", data_size, data_size/PARALLEL_FILTER);
        merlinL75:
        for (int j = 0; j < data_size / PARALLEL_FILTER; j++) {
            #pragma HLS pipeline
            #pragma HLS loop_tripcount min=21632 max=21632
            ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > tmp = stream_input . read();
            stream_output . write(tmp);
    #ifdef DEBUG_SHORTCUT
            int out_w = config_list[5];
            int p = 0;
            ap_int< 8 > tmp_buf = (tmp(((p + 1) * 8 - 1), (p * 8)));
            printf("debug_shortcut[%3d][%3d]=%d ", j / out_w,j % out_w,(tmp_buf . to_int()));
            if (j % out_w == out_w - 1) {
                printf("\n");
            }
    #endif
        }
    }
    #ifdef DEBUG_SHORTCUT
    printf("finish shortcut \n");
    #endif
}

void upsample_core(
    hls::stream< ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > > &input_data, 
    ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > tmp_line[2*56],
    int in_w) 
{
    int index = in_w * 2;
    for (int i = 0; i < in_w; ++i) {
        #pragma HLS loop_tripcount min=28 max=28
        ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > tmp = input_data . read();
        tmp_line[i * 2] = tmp;
        tmp_line[i * 2 + 1] = tmp;
        tmp_line[index + i * 2] = tmp;
        tmp_line[index + i * 2 + 1] = tmp;
    }
}

void upsample_out(
    ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > tmp_line[2*56], 
    hls::stream< ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > > &output_data,
    int out_w) 
{
    for (int i = 0; i < 2 * out_w; i++) {
        #pragma HLS loop_tripcount min=112 max=112
        ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > out_buf = tmp_line[i];
        output_data . write(out_buf);
    }
}

void upsample_stream(
    hls::stream< ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > > &input_data, 
    hls::stream< ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > > &output_data,
    const int config_list[32])
{
    #pragma HLS inline off
    //printf("start upsample steaming\n");
    int in_w = config_list[1];
    int in_h = config_list[2];
    int in_c = config_list[3];
    int out_w = config_list[5];
    ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > ping[2*56];
    #pragma HLS resource variable=ping core=RAM_1P_LUTRAM
    ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > pong[2*56];
    #pragma HLS resource variable=pong core=RAM_1P_LUTRAM
    for (int k = 0; k < in_c / PARALLEL_FILTER; ++k) {
        #pragma HLS loop_tripcount min=16 max=16
        int j = 0;
        upsample_core(input_data, ping, in_w);
        for (j = 0; j < in_h - 1; ++j) {
            #pragma HLS loop_tripcount min=26 max=26
            if(j % 2 == 0) {
                upsample_core(input_data, pong, in_w);
                upsample_out(ping, output_data, out_w);
            } else {
                upsample_core(input_data, ping, in_w);
                upsample_out(pong, output_data, out_w);
            }
        }
        if(j % 2 == 0) {
            upsample_out(ping, output_data, out_w);
        } else {
            upsample_out(pong, output_data, out_w);
        }
    }
}

void upsample_switch(
    hls::stream< ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > > &data_input, 
    hls::stream< ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > > &data_output, 
    const int config_list[32]) 
{
    #pragma HLS inline off
    #ifdef DEBUG_UPSAMPLE
    printf("start upsample \n");
    #endif
    int upsample_sel = config_list[0];
    if (upsample_sel == 1) {
        upsample_stream(data_input, data_output, config_list);
    } else {
        int data_size = config_list[19];
        //int data_size = config_list[5] * config_list[6] * config_list[7];
        //printf("upsample data size %d %d\n", data_size, data_size/PARALLEL_FILTER);
        for (int j = 0; j < data_size / PARALLEL_FILTER; j++) {
            #pragma HLS pipeline
            #pragma HLS loop_tripcount min=21632 max=21632
            ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > tmp = data_input . read();
            data_output . write(tmp);
#ifdef DEBUG_UPSAMPLE
            int out_w = config_list[5];
            for(int p = 0; p < PARALLEL_FILTER; p++){
                ap_int< 8 > tmp_buf = (tmp(((p + 1) * 8 - 1), (p * 8)));
                printf("[p%d] debug_upsample[%3d][%3d]=%d ", p, j / out_w,j % out_w,(tmp_buf . to_int()));
            }
            printf("\n");
#endif
        }
    }
    #ifdef DEBUG_UPSAMPLE
    printf("finish upsample \n");
    #endif
}

void read_fifo(
    hls::stream< ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > > &stream_data_in, 
    int read_fifo_line, int out_w, int new_w,int out_h, int out_h_4,
    ap_int< WIDE_BUS_WIDTH > *data_out)
{
    //int burst_length = 13 * 416;
    //if(out_w == 56)
    //    burst_length = 13 * out_w * 8;
    //else if(out_w == 28 || out_w == 26 || out_w == 32 || out_w == 16 || out_w == 14)
    //    burst_length = 13 * out_w * 16;
    //int out_h_4 = out_h % FACTORS;

    ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > tmp_buf[4];
    ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > tmp_buf2[4];
    merlinL89:
    for (int i = 0; i < read_fifo_line; i++) {
        #pragma hls loop_tripcount min=208 max=208
        //printf("read_fifo:i %d, new_w=%d, out_w=%d, out_w/4=%d\n",i,new_w, out_w, out_w / 4);
        merlinL88:
        for (int j = 0; j < out_h / FACTORS; j++) {
            #pragma hls loop_tripcount min=104 max=104
            tmp_buf[0] = stream_data_in . read();
            tmp_buf[1] = stream_data_in . read();
            tmp_buf[2] = stream_data_in . read();
            tmp_buf[3] = stream_data_in . read();

            data_out[i * new_w / FACTORS + j] = (((tmp_buf[3] , tmp_buf[2] , tmp_buf[1])) , tmp_buf[0]);

    #ifdef DEBUG_DATAOUT
            int p = 0;
            printf("index[%3d] ", i * new_w / 4 + j);
            ap_int< 8 > tmp0_buf = (tmp_buf[0](((p + 1) * 8 - 1), (p * 8)));
            printf("debug1_image_out[%3d][%3d]=%d ", i%out_h, j*4+0, (tmp0_buf . to_int()));
            ap_int< 8 > tmp1_buf = (tmp_buf[1](((p + 1) * 8 - 1), (p * 8)));
            printf("debug2_image_out[%3d][%3d]=%d ", i%out_h, j*4+1, (tmp1_buf . to_int()));
            ap_int< 8 > tmp2_buf = (tmp_buf[2](((p + 1) * 8 - 1), (p * 8)));
            printf("debug3_image_out[%3d][%3d]=%d ", i%out_h, j*4+2, (tmp2_buf . to_int()));
            ap_int< 8 > tmp3_buf = (tmp_buf[3](((p + 1) * 8 - 1), (p * 8)));
            printf("debug4_image_out[%3d][%3d]=%d ", i%out_h, j*4+3, (tmp3_buf . to_int()));
    #endif
        }
        if(out_h_4 != 0) {
            for(int j = 0; j < FACTORS; j++){
                #pragma hls loop_tripcount min=4 max=4
                tmp_buf[j] = 0;
            }
            for(int j = 0; j < out_h_4; j++){
                #pragma hls loop_tripcount min=4 max=4
                tmp_buf[j] = stream_data_in . read();
            }
            data_out[i * new_w / FACTORS + out_h / FACTORS] = (((tmp_buf[3] , tmp_buf[2] , tmp_buf[1])) , tmp_buf[0]);
    #ifdef DEBUG_DATAOUT
            printf("index[%3d] ", i * new_w / 4 + out_h / 4);
            int p = 0;
            ap_int< 8 > tmp0_buf = (tmp_buf[0](((p + 1) * 8 - 1), (p * 8)));
            printf("debug0_image_out[%3d][%3d]=%d ", i%out_h, (out_h / 4)*4+0, (tmp0_buf . to_int()));
            ap_int< 8 > tmp1_buf = (tmp_buf[1](((p + 1) * 8 - 1), (p * 8)));
            printf("debug0_image_out[%3d][%3d]=%d ", i%out_h, (out_h / 4)*4+1, (tmp1_buf . to_int()));
            ap_int< 8 > tmp2_buf = (tmp_buf[2](((p + 1) * 8 - 1), (p * 8)));
            printf("debug0_image_out[%3d][%3d]=%d ", i%out_h, (out_h / 4)*4+2, (tmp2_buf . to_int()));
            ap_int< 8 > tmp3_buf = (tmp_buf[3](((p + 1) * 8 - 1), (p * 8)));
            printf("debug0_image_out[%3d][%3d]=%d ", i%out_h, (out_h / 4)*4+3, (tmp3_buf . to_int()));
    #endif
        }
        if(out_w != out_h) {
            for(int j = 0; j < out_w - out_h; j++){
                #pragma hls loop_tripcount min=3 max=3
                stream_data_in . read();
            }
        }
    #ifdef DEBUG_DATAOUT
        printf("\n");
    #endif
    }
}


void stream_out_image(
    hls::stream< ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > > &stream_data_in, 
    ap_int< WIDE_BUS_WIDTH > *data_out, 
    const int config_list[32])
{
    #pragma HLS inline off
    printf("start stream out image \n");
    int out_w = config_list[5];
    int out_h = config_list[6];
    int new_w = config_list[8];
    int read_fifo_line = config_list[20];
    int burst_length = config_list[24];
    int loop_bound = config_list[25];
    int out_h_4 = config_list[26];
    int output_idx = config_list[29];
    //printf("out_w:%d, new_w:%d, out_h:%d\n", out_w, new_w, out_h);
    //printf("output_idx:%d\n", output_idx);
    //printf("stream_out_image-0:burst_length = %d\n", burst_length);
    //printf("stream_out_image-0:loop_bound = %d\n", loop_bound);
    //if(out_w == 56)
    //    burst_length = 13 * new_w * 8 / 4;
    //else if(new_w == 28)
    //    burst_length = 13 * new_w * 16 / 4;
    //else if(new_w == 16 || out_w == 14)
    //    burst_length = 13 * new_w * 16 / 4;
    //else
    //    burst_length = 13 * 416 / 4;

    //int data_size = config_list[8] * config_list[6] * config_list[7];
    //loop_bound = data_size / 16 / burst_length;
    //loop_bound = data_size / PARALLEL_FILTER / FACTORS / burst_length;
    //printf("stream_out_image-1:burst_length = %d\n", burst_length);
    //printf("stream_out_image-1:loop_bound = %d\n", loop_bound);

    ap_int< WIDE_BUS_WIDTH > buf_ping_data_out[13*128];// max of 13*416/4 13*52*2 13*28*4 13*16*8
    ap_int< WIDE_BUS_WIDTH > buf_pang_data_out[13*128];// max of 13*416/4 13*52*2 13*28*4 13*16*8
    read_fifo(stream_data_in, read_fifo_line, out_w, new_w, out_h, out_h_4, buf_ping_data_out);
    int index = output_idx;
    merlinL86:
    for (int j = 0; j < loop_bound - 1; j++) {
        #pragma HLS loop_tripcount min=128 max=128
        if (j % 2 == 0) {
            memcpy((void*)&data_out[index], (void*)buf_ping_data_out, sizeof(ap_int< WIDE_BUS_WIDTH >) * burst_length);
            read_fifo(stream_data_in, read_fifo_line, out_w, new_w, out_h, out_h_4, buf_pang_data_out);
        }
        else {
            memcpy((void*)&data_out[index], (void*)buf_pang_data_out, sizeof(ap_int< WIDE_BUS_WIDTH >) * burst_length);
            read_fifo(stream_data_in, read_fifo_line, out_w, new_w, out_h, out_h_4, buf_ping_data_out);
        }
        index += burst_length;
    }
    if (loop_bound % 2 == 0) {
        memcpy((void*)&data_out[index], (void*)buf_pang_data_out, sizeof(ap_int< WIDE_BUS_WIDTH >) * burst_length);
    }
    else {
        memcpy((void*)&data_out[index], (void*)buf_ping_data_out, sizeof(ap_int< WIDE_BUS_WIDTH >) * burst_length);
    }
    //printf("finish_stream out image\n");
}

#pragma ACCEL kernel
void top_kernel(ap_int< WIDE_BUS_WIDTH > *merlin_input, 
                ap_int< WIDE_BUS_WIDTH > *input_add, 
                ap_int< WIDE_BUS_WIDTH > *merlin_output, 
                ap_int< WIDE_BUS_WIDTH > *weights, 
                int layer_min, int layer_max)
{
    #pragma ACCEL interface variable=merlin_input   depth=173056 max_depth=173056 bundle = gmem0
    #pragma ACCEL interface variable=input_add      depth=173056 max_depth=173056 bundle = gmem1
    #pragma ACCEL interface variable=merlin_output  depth=172380 max_depth=172380 bundle = gmem1
    #pragma ACCEL interface variable=weights        depth=262144 max_depth=262144 bundle = gmem2
    
    printf("start kernel !!! \n");
    hls::stream< ap_int< WIDE_BUS_WIDTH > > fifo_image_1x1("fifo_image_1x1");
    #pragma HLS stream variable = fifo_image_1x1 depth = 512
    #pragma HLS resource variable = fifo_image_1x1 core=fifo_lutram

    hls::stream< ap_int< WIDE_BUS_WIDTH > > fifo_image_ping("fifo_image_ping");
    #pragma HLS stream variable = fifo_image_ping depth = 512
    #pragma HLS resource variable = fifo_image_ping core=fifo_lutram

    hls::stream< ap_int< WIDE_BUS_WIDTH > > fifo_image_pang("fifo_image_pang");
    #pragma HLS stream variable = fifo_image_pang depth = 512
    #pragma HLS resource variable = fifo_image_pang core=fifo_lutram

    hls::stream< CONV_DT > fifo_image_conv[PARALLEL_FILTER];
    #pragma HLS stream variable = fifo_image_conv depth = 512

    hls::stream< ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > > fifo_image_bias("fifo_image_bias");
    #pragma HLS stream variable = fifo_image_bias depth = 512

    hls::stream< ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > > fifo_image_shortcut("fifo_image_shortcut");
    #pragma HLS stream variable = fifo_image_shortcut depth = 512

    hls::stream< ap_int< ORG_DATA_WIDTH*PARALLEL_FILTER > > fifo_image_upsample("fifo_image_upsample");
    #pragma HLS stream variable = fifo_image_upsample depth = 512

    hls::stream< ap_int< WIDE_BUS_WIDTH > > fifo_bias("fifo_bias");
    #pragma HLS stream variable = fifo_bias depth = 512

    hls::stream< ap_int< WIDE_BUS_WIDTH > > fifo_weights_3x3("fifo_weights_3x3");
    #pragma HLS resource variable = fifo_weights_3x3 core=fifo_lutram
    #pragma HLS stream variable = fifo_weights_3x3 depth = 512

    hls::stream< ap_int< WIDE_BUS_WIDTH > > fifo_weights_1x1("fifo_weights_1x1");
    #pragma HLS resource variable = fifo_weights_1x1 core=fifo_lutram
    #pragma HLS stream variable = fifo_weights_1x1 depth = 512

    merlinL94:
    for (int layer_cnt = layer_min; layer_cnt < layer_max + 1; layer_cnt++) {

        #pragma HLS loop_tripcount min=1 max=1
        printf("layer %d to %d : %d\n", layer_min,layer_max,layer_cnt);

        #pragma HLS dataflow
        stream_in_weights(
            weights, 
            fifo_weights_3x3, 
            fifo_weights_1x1, 
            fifo_bias, 
            config_list_all[layer_cnt][0]);

        stream_in_image(
            merlin_input, 
            fifo_image_ping, 
            fifo_image_pang, 
            fifo_image_1x1, 
            config_list_all[layer_cnt][0]);

        conv_switch(
            fifo_image_ping, 
            fifo_image_pang, 
            fifo_image_1x1, 
            fifo_weights_3x3, 
            fifo_weights_1x1, 
            fifo_image_conv, 
            config_list_all[layer_cnt][0]);

        bias_switch(
            fifo_image_conv, 
            fifo_bias, 
            fifo_image_bias, 
            config_list_all[layer_cnt][0]);

        shortcut_switch(
            input_add, 
            fifo_image_bias, 
            fifo_image_shortcut, 
            config_list_all[layer_cnt][1]);

        upsample_switch(
            fifo_image_shortcut, 
            fifo_image_upsample, 
            config_list_all[layer_cnt][2]);

        stream_out_image(
            fifo_image_upsample, 
            merlin_output, 
            config_list_all[layer_cnt][2]);
    }
    printf("finish kernel !!! \n");
}
