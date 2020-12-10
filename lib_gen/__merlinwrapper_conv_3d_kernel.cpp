#include <string.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include <hls_stream.h>
#include "__merlinhead_kernel_top.h"
#include <chrono> 
#include <iostream>
#include "__merlin_opencl_if.h"
#include "time.h"
#include <sys/time.h>
//#define DEBUG_LIB

void write_data_file_float(int layer, float * value, int size) {
    char file_name[] = "output_layerXXX.dat";
    file_name[12] = layer/100 + '0';
    file_name[13] = (layer%100)/10 + '0';
    file_name[14] = (layer%100)%10 + '0';
    printf("Output data name = %s\n", file_name);
    // write file
    FILE* fp2= fopen(file_name, "w+");
    if (NULL == fp2) {
        fclose(fp2);
        //exit(1);
    }
    int i = 0;
    for(i=0; i<size; i++) {
        fprintf(fp2, "%f\n", value[i]);
    }
    fclose(fp2);
    //exit(1);
}

int entry_index_s(int location, int entry, int size, int classes)
{
    int n =   location / size;
    int loc = location % size;
    return n*size*(4+classes+1) + entry*size + loc;
}

void activate_array_s(float *x, const int n)
{
    int i;
    for(i = 0; i < n; ++i){
        x[i] = 1./(1. + exp(-x[i]));
    }
}

void yolo_layer_q(int8_t *input, float *yolo_out, int outputs, int size) {
    int i,j,t,n,i_q;
    int classes = 20;
    int n_max = 3;
//    printf("outputs=%d, n=%d, size=%d, classes=%d\n", outputs, n_max, size, classes);
    for(i_q = 0; i_q < outputs;  ++ i_q) 
        yolo_out[i_q] = input[i_q] / 4.0;
    for(n = 0; n < n_max; ++n){
        int index = entry_index_s(n*size, 0, size, classes);
        activate_array_s(yolo_out + index, 2*size);
        index = entry_index_s(n*size, 4, size, classes);
        activate_array_s(yolo_out + index, (1+classes)*size);
    }
}

double what_time_is_it_now()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
// if tile = 8
// A0B0C0D0E0F0G0H0......
// I0J0K0L0M0N0O0P0......
// ......
// =========>
// A0A1A2A3A4A5A6A7....
// B0B1B2B3B4B5B6B7....
// ......
// H0H1H2H3H4H5H6H7....
// ......
void data_format_transform_back(DATA_T *a, DATA_T *b, int *config) {
//    printf("enter data transformation back\n");
    int w = config[0];
    int h = config[1];
    int c_in = config[2];
    int c_out = config[3];
    int tile = config[4];
    int input_total_size = h * w * c_in;
    int output_total_size = h * h * c_out;
    int image_size = h * w;
    int one_tile_size = h * w * tile;
    //printf("h = %d, w = %d, c_in = %d, c_out = %d, tile = %d\n", h, w, c_in, c_out, tile);
    //printf("input_total_size = %d, output_total_size = %d, image_size = %d, one_tile_size = %d\n", input_total_size, output_total_size, image_size, one_tile_size);
    int i, j, k, i_sub;
    for (i = 0; i < c_out/tile; i++) {
        int total_tile_size = i * tile * h * w;
        for (i_sub = 0; i_sub < tile; i_sub++) {
            for (j = 0; j < h; j++) {
                for (k = 0; k < h; k++) {
                    int index1 = i_sub * h * w + j * w + k;
                    int output_index = i * tile * h * h + i_sub * h * h + j * h + k;
                    int input_index = total_tile_size + (index1 % image_size) * tile + index1 / image_size;
                    DATA_T tmp;
                    if (input_index >= input_total_size) {
                        tmp = 0;
                    } else {
                        tmp = a[input_index];
                    }
                    //printf("i %d, i_sub %d, j %d, k %d, index1 = %d, output_index = %d, input_index = %d, tmp = %d\n", i, i_sub, j, k,index1, output_index, input_index, tmp);
                    b[output_index] = tmp;
                }
            }
        }
    }
//    FILE *fp_out;
//    char * output_file = "reformat_back.dat";
//    fp_out = fopen(output_file, "w");
//    if(fp_out == NULL) {
//        perror("fopen");
//        return;
//    }
//    for(i=0; i<output_total_size; i++) {
//        fprintf(fp_out, "%f\n", b[i]);
//    }
//    fclose(fp_out);
}

// if tile = 8
// A0A1A2A3A4A5A6A7....
// B0B1B2B3B4B5B6B7....
// ......
// H0H1H2H3H4H5H6H7....
// ......
// =========>
// A0B0C0D0E0F0G0H0......
// I0J0K0L0M0N0O0P0......
// ......
void data_format_transform(DATA_T *a, DATA_T *b, int *config) {
//    printf("enter data transformation\n");
    int w = config[0];
    int h = config[1];
    int c_in = config[2];
    int c_out = config[3];
    int tile = config[4];
    int input_total_size = h * w * c_in;
    int output_total_size = h * w * c_out;
    int image_size = h * w;
    int one_tile_size = h * w * tile;
    //printf("h = %d, w = %d, c_in = %d, c_out = %d, tile = %d\n", h, w, c_in, c_out, tile);
    //printf("input_total_size = %d, output_total_size = %d, image_size = %d, one_tile_size = %d\n", input_total_size, output_total_size, image_size, one_tile_size);
    int i, j, k, i_sub;
    DATA_T * a_tmp = (DATA_T*)malloc(w*h*c_in * sizeof(DATA_T));
    for (i = 0; i < c_in; i++) {
        for (j = 0; j < h; j++) {
            for (k = 0; k < h; k++) {
                a_tmp[i*h*w + j*w + k] = a[i*h*h + j*h + k];
                //printf("in_tmp[%d][%d][%d] = %d ", i, j, k, a_tmp[i*h*w + j*w + k]);
            }
            for (k = h; k < w; k++) {
                a_tmp[i*h*w + j*w + k] = 0;
                //printf("in_tmp[%d][%d][%d] = %d ", i, j, k, 0);
            }
            //printf("\n");
        }
    }
    for (i = 0; i < c_out/tile; i++) {
        int total_tile_size = i * tile * h * w;
        for (i_sub = 0; i_sub < tile; i_sub++) {
            for (j = 0; j < h; j++) {
                for (k = 0; k < w; k++) {
                    int index1 = i_sub * h * w + j * w + k;
                    int output_index = i * tile * h * w + i_sub * h * w + j * w + k;
                    int input_index = total_tile_size + (index1 % tile) * image_size + index1 / tile;
                    DATA_T tmp;
                    if (input_index >= input_total_size) {
                        tmp = 0;
                    } else {
                        tmp = a_tmp[input_index];
                    }
                    //printf("i %d, i_sub %d, j %d, k %d, index1 = %d, output_index = %d, input_index = %d, tmp = %d\n", i, i_sub, j, k,index1, output_index, input_index, tmp);
                    b[output_index] = tmp;
                }
            }
        }
    }
}

void write_data_file(int layer, DATA_T * value, int size) {
    char file_name[] = "output_layerXXX.dat";
    file_name[12] = layer/100 + '0';
    file_name[13] = (layer%100)/10 + '0';
    file_name[14] = (layer%100)%10 + '0';
    printf("Output data name = %s\n", file_name);
    // write file
    FILE* fp2= fopen(file_name, "w+");
    if (NULL == fp2) {
        fclose(fp2);
        //exit(1);
    }
    int i = 0;    
    for(i=0; i<size; i++) {
        fprintf(fp2, "%d\n", value[i]);
    }
    fclose(fp2);
    //exit(1);
}


//===========================================//
//use pingpong queue to do batches of images
//when ping do computing, pong do data transfor and some other data processes
//===========================================//  
int __merlin_exec_top_kernel_overlap(DATA_T * input,
                                     float * yolo1_out, float * yolo2_out, float * yolo3_out,
                                     int batch,
                                     int * debug_config)
{
    int i=0;
    int debug_layer = debug_config[0];
    int old_layer = debug_config[1];
    int layer_min = debug_config[2];
    int layer_max = debug_config[3];
    int layer_size = debug_config[4];
    int layer_c = debug_config[5];
    int image_size = config_list_all[layer_min][0][1] * config_list_all[layer_min][0][2] * layer_c;
#ifdef DEBUG_LIB
    printf("debug value:new_layer=%d, old_layer=%d, min=%d, max=%d, size=%d, c=%d\n", debug_config[0], debug_config[1], debug_config[2], debug_config[3], debug_config[4], debug_config[5]);
    printf("input size:%d %d\n", layer_size, config_list_all[layer_min][0][17]);
    printf("input index:%d\n", config_list_all[layer_min][0][28]);
    printf("w:%d, h:%d, c:%d\n", config_list_all[layer_min][0][1], config_list_all[layer_min][0][2], layer_c);
#endif

    int config_format_out[5];
    int config_format_in[5];
    config_format_in[0] = config_list_all[layer_min][0][1];
    config_format_in[1] = config_list_all[layer_min][0][2];
    config_format_in[2] = layer_c;
    config_format_in[3] = ((layer_c + PARALLEL_FILTER - 1)/PARALLEL_FILTER)*PARALLEL_FILTER;
    config_format_in[4] = PARALLEL_FILTER;
    DATA_T * layer_0_in_format[2]; 
    for(i=0; i<OVERLAP; i++) {
        layer_0_in_format[i] = (DATA_T*)malloc(config_list_all[layer_min][0][17] * sizeof(DATA_T));
    }
#ifdef DEBUG_LIB
    double time;
#endif
    int overlap = batch == 1 ? 1 : OVERLAP;
    int layer_cnt = 0;
    int flag = 0;
    int frame_cnt = 0;
    int queue_idx = 0;
//    printf("first layer %f seconds.\n", what_time_is_it_now()); 

#ifdef SOC
    int8_t* data[2];
    for(i=0; i<OVERLAP; i++) {
        data[i] = (int8_t *)q[i]->enqueueMapBuffer(*buffer_input[i],
                                               CL_TRUE,
                                               CL_MAP_WRITE | CL_MAP_READ,
                                               0,
                                               sizeof(int8_t) * OUTPUT_LAYER_NUM*1024*1024*sizeof(DATA_T));
    }
#else
#endif // SOC

    for(frame_cnt = 0; frame_cnt < batch + overlap; frame_cnt++){
        queue_idx = overlap == 1 ? 0 : flag % OVERLAP;
        if(frame_cnt >= overlap){
            //===========================================//
            //wait queue to finish
            //ping wait last ping task to finish
            //pong wait last pong task to finish
            //===========================================//
            q[queue_idx]->finish();
//            printf("layer %f seconds.\n", what_time_is_it_now()); 
#ifdef DEBUG_LIB
            printf("queue %d finished\n", queue_idx);
#endif
            // copy 3 yolo layer out
            int size1 = config_list_all[58][2][8] * config_list_all[58][2][6] * config_list_all[58][2][7];
            int size2 = config_list_all[66][2][8] * config_list_all[66][2][6] * config_list_all[66][2][7];
            int size3 = config_list_all[74][2][8] * config_list_all[74][2][6] * config_list_all[74][2][7];
            //===========================================//
            //memcpy yolo layer input data from global memory offset
            //===========================================//
#ifdef DEBUG_LIB
            time=what_time_is_it_now();
#endif
            DATA_T * yolo1_pre_format = (DATA_T *)malloc(sizeof(DATA_T) * size1); 
            DATA_T * yolo2_pre_format = (DATA_T *)malloc(sizeof(DATA_T) * size2); 
            DATA_T * yolo3_pre_format = (DATA_T *)malloc(sizeof(DATA_T) * size3); 
#ifdef SOC
            memcpy(yolo1_pre_format, data[queue_idx] + config_list_all[58][2][29]*WIDE_BUS_WIDTH/ORG_DATA_WIDTH, size1 * sizeof(DATA_T));
            memcpy(yolo2_pre_format, data[queue_idx] + config_list_all[66][2][29]*WIDE_BUS_WIDTH/ORG_DATA_WIDTH, size2 * sizeof(DATA_T));
            memcpy(yolo3_pre_format, data[queue_idx] + config_list_all[74][2][29]*WIDE_BUS_WIDTH/ORG_DATA_WIDTH, size3 * sizeof(DATA_T));
#else
            memcpy(yolo1_pre_format, data_input[queue_idx].data() + config_list_all[58][2][29]*WIDE_BUS_WIDTH/ORG_DATA_WIDTH, size1 * sizeof(DATA_T));
            memcpy(yolo2_pre_format, data_input[queue_idx].data() + config_list_all[66][2][29]*WIDE_BUS_WIDTH/ORG_DATA_WIDTH, size2 * sizeof(DATA_T));
            memcpy(yolo3_pre_format, data_input[queue_idx].data() + config_list_all[74][2][29]*WIDE_BUS_WIDTH/ORG_DATA_WIDTH, size3 * sizeof(DATA_T));
#endif // SOC
#ifdef DEBUG_LIB
            printf("last layer memcpy out %f seconds.\n", what_time_is_it_now()-time); 
#endif

            //===========================================//
            //change data alignment for CPU process
            //===========================================//
#ifdef DEBUG_LIB
            time=what_time_is_it_now();
#endif
            DATA_T * yolo1_pre = (DATA_T *)malloc(sizeof(DATA_T) * config_list_all[58][2][8] * config_list_all[58][2][6] * config_list_all[58][2][7]);
            DATA_T * yolo2_pre = (DATA_T *)malloc(sizeof(DATA_T) * config_list_all[66][2][8] * config_list_all[66][2][6] * config_list_all[66][2][7]);
            DATA_T * yolo3_pre = (DATA_T *)malloc(sizeof(DATA_T) * config_list_all[74][2][8] * config_list_all[74][2][6] * config_list_all[74][2][7]);
            config_format_out[0] = 16;
            config_format_out[1] = 13;
            config_format_out[2] = 80;
            config_format_out[3] = 80;
            config_format_out[4] = PARALLEL_FILTER;
            data_format_transform_back(yolo1_pre_format, yolo1_pre, config_format_out);
            config_format_out[0] = 28;
            config_format_out[1] = 26;
            config_format_out[2] = 80;
            config_format_out[3] = 80;
            config_format_out[4] = PARALLEL_FILTER;
            data_format_transform_back(yolo2_pre_format, yolo2_pre, config_format_out);
            config_format_out[0] = 52;
            config_format_out[1] = 52;
            config_format_out[2] = 80;
            config_format_out[3] = 80;
            config_format_out[4] = PARALLEL_FILTER;
            data_format_transform_back(yolo3_pre_format, yolo3_pre, config_format_out);
#ifdef DEBUG_LIB
            printf("data transform back %f seconds.\n", what_time_is_it_now()-time); 
#endif
            //===========================================//
            //run yolo layer
            //===========================================//
#ifdef DEBUG_LIB
            time=what_time_is_it_now();
#endif
            //TODO: in batch mode, it need to add offset for yolo outs
            yolo_layer_q(yolo1_pre, yolo1_out + (frame_cnt - overlap) * 12675, 12675,  13*13);
            yolo_layer_q(yolo2_pre, yolo2_out + (frame_cnt - overlap) * 50700, 50700,  26*26);
            yolo_layer_q(yolo3_pre, yolo3_out + (frame_cnt - overlap) * 202800, 202800, 52*52);
#ifdef DEBUG_LIB
            printf("3 yolos in %f seconds.\n", what_time_is_it_now()-time); 
#endif
#ifdef DEBUG_LIB
            write_data_file_float(82 + 100*(frame_cnt - overlap), yolo1_out + (frame_cnt - overlap) * 12675,  12675);
            write_data_file_float(94 + 100*(frame_cnt - overlap), yolo2_out + (frame_cnt - overlap) * 50700,  50700);
            write_data_file_float(106+ 100*(frame_cnt - overlap), yolo3_out + (frame_cnt - overlap) * 202800, 202800);
            printf("finish yolo-1\n");
            printf("finish yolo-2\n");
            printf("finish yolo-3\n");
#endif

#ifdef DEBUG_LIB
            // debug information
            printf("saving layer...\n");
            for(int layer_x = layer_min; layer_x < layer_max + 1; layer_x++){
                printf("saving layer %d size %d index:%d\n", layer_x, config_list_all[layer_x][2][19], config_list_all[layer_x][2][29]);
                int size_x = config_list_all[layer_x][2][8] * config_list_all[layer_x][2][6] * config_list_all[layer_x][2][7];
                DATA_T * layer_x_out_format = (DATA_T *)malloc(sizeof(DATA_T) * size_x); 
#ifdef SOC
                memcpy(layer_x_out_format, data[queue_idx] + config_list_all[layer_x][2][29]*WIDE_BUS_WIDTH/ORG_DATA_WIDTH, size_x * sizeof(DATA_T));
#else
                memcpy(layer_x_out_format, data_input[queue_idx].data() + config_list_all[layer_x][2][29]*WIDE_BUS_WIDTH/ORG_DATA_WIDTH, size_x * sizeof(DATA_T));
#endif //SOC
                config_format_in[0] = config_list_all[layer_x][2][8];
                config_format_in[1] = config_list_all[layer_x][2][6];
                config_format_in[2] = config_list_all[layer_x][2][7];
                config_format_in[3] = config_list_all[layer_x][2][7];
                config_format_in[4] = PARALLEL_FILTER;
                DATA_T * layer_x_out = (DATA_T *)malloc(sizeof(DATA_T) * size_x);
                data_format_transform_back(layer_x_out_format, layer_x_out, config_format_in);
                int org_layer;
                int data_size = config_list_all[layer_x][2][6]*config_list_all[layer_x][2][6]*config_list_all[layer_x][2][7];
                if(layer_x == 58){
                    org_layer = 81;// old layer
                    data_size = config_list_all[layer_x][2][6]*config_list_all[layer_x][2][6]*75;
                } else if(layer_x == 59){
                    org_layer = 85;// old layer
                } else if(layer_x == 66){
                    org_layer = 93;// old layer
                    data_size = config_list_all[layer_x][2][6]*config_list_all[layer_x][2][6]*75;
                } else if(layer_x == 67){
                    org_layer = 97;// old layer
                } else if(layer_x == 74){
                    org_layer = 105;// old layer
                    data_size = config_list_all[layer_x][2][6]*config_list_all[layer_x][2][6]*75;
                } else{
                    org_layer = index_conv[layer_x + 1] - 1;
                }
                //write_data_file(org_layer+500, layer_x_out_format, data_size);//debug layer
                write_data_file(org_layer, layer_x_out, data_size);//debug layer
            }
#endif
        }
        if(frame_cnt < batch){
            //===========================================//
            //realign image data for accelerate
            //===========================================//
            //write_data_file(202, input, config_list_all[layer_min][0][17]);//debug layer
#ifdef DEBUG_LIB
            time=what_time_is_it_now();
#endif
            //TODO: in batch mode, it need to add offset for input images
            int offset = frame_cnt * (416 * 416 *3); //frame_cnt * image_size;
            data_format_transform(input + offset, layer_0_in_format[queue_idx], config_format_in);
            //data_format_transform(input, layer_0_in_format[queue_idx], config_format_in);
#ifdef DEBUG_LIB
            write_data_file(0 + frame_cnt * 100, layer_0_in_format[queue_idx], config_list_all[layer_min][0][17]);//debug layer
            printf("first layer data transform in %f seconds.\n", what_time_is_it_now()-time); 
#endif

            //===========================================//
            //copy input data to alignment buffer
            //===========================================//
#ifdef DEBUG_LIB
            time=what_time_is_it_now();
#endif
#ifdef SOC
            memcpy(data[queue_idx] + config_list_all[layer_min][0][28]*WIDE_BUS_WIDTH/ORG_DATA_WIDTH, 
                    layer_0_in_format[queue_idx], 
                    config_list_all[layer_min][0][17] * sizeof(DATA_T));
#else
            memcpy(data_input[queue_idx].data() + config_list_all[layer_min][0][28]*WIDE_BUS_WIDTH/ORG_DATA_WIDTH, 
                    layer_0_in_format[queue_idx], 
                    config_list_all[layer_min][0][17] * sizeof(DATA_T));
#endif // SOC
#ifdef DEBUG_LIB
            printf("first layer memcpy in %f seconds.\n", what_time_is_it_now()-time); 
#endif
        }
        if(frame_cnt < batch){
            //===========================================//
            // copy buffer and execute kernel
            //===========================================//
#ifdef DEBUG_LIB
            time=what_time_is_it_now();
#endif
            q[queue_idx]->enqueueMigrateMemObjects({*(buffer_input[queue_idx])}, 0);
#ifdef DEBUG_LIB
            q[queue_idx]->finish();
            printf("migrate in %f seconds.\n", what_time_is_it_now()-time); 
#endif
            top_kernel->setArg(0, *buffer_input[queue_idx]);
            top_kernel->setArg(1, *buffer_input[queue_idx]);
            top_kernel->setArg(2, *buffer_input[queue_idx]);
            top_kernel->setArg(3, *buffer_weights[queue_idx]);
            top_kernel->setArg(4, layer_min);
            top_kernel->setArg(5, layer_max);
            q[queue_idx]->enqueueTask(*top_kernel);
#ifdef DEBUG_LIB
            q[queue_idx]->finish();
            time=what_time_is_it_now();
#endif
            q[queue_idx]->enqueueMigrateMemObjects({*(buffer_input[queue_idx])}, CL_MIGRATE_MEM_OBJECT_HOST);
#ifdef DEBUG_LIB
            q[queue_idx]->finish();
            printf("migrate out %f seconds.\n", what_time_is_it_now()-time); 
#endif
        }
        flag++;
    }
    return 0;
}

//===========================================//
//write weight and bias data to pingpong queue
//then write to global memory only once
//===========================================//  
int __merlin_load_weight(DATA_T *weights[75], int32_t bias[75][1024]) {
    int layer_cnt = 0;
#ifdef DEBUG_LIB
    printf("memcpy weight to alignment buffer\n");
#endif
#ifdef SOC
    int8_t* weight_in[2] ;
    for(int i=0; i<OVERLAP; i++) {
        weight_in[i] = (int8_t *)q[i]->enqueueMapBuffer(*buffer_weights[i],
                                                 CL_TRUE,
                                                 CL_MAP_WRITE,
                                                 0,
                                                 sizeof(int8_t) * (OUTPUT_LAYER_NUM*1024*1024 + OUTPUT_LAYER_NUM*1024*sizeof(BIAS_DT)));
    }
    for(layer_cnt = 0; layer_cnt < OUTPUT_LAYER_NUM; layer_cnt++){
        for(int i=0; i<OVERLAP; i++) {
            // copy weight
            memcpy(weight_in[i] + config_list_all[layer_cnt][0][30]*WIDE_BUS_WIDTH/ORG_DATA_WIDTH,
                   weights[layer_cnt],
                   config_list_all[layer_cnt][0][0] * config_list_all[layer_cnt][0][0] \
                   * config_list_all[layer_cnt][0][3] \
                   * config_list_all[layer_cnt][0][7] \
                   * sizeof(DATA_T));
            // copy bias
            memcpy(weight_in[i] + OUTPUT_LAYER_NUM * 1024 * 1024 + layer_cnt * 1024 * sizeof(BIAS_DT),
                   bias[layer_cnt],
                   config_list_all[layer_cnt][0][7] * sizeof(BIAS_DT));
        }
    }
#else
    for(layer_cnt = 0; layer_cnt < OUTPUT_LAYER_NUM; layer_cnt++){
        for(int i=0; i<OVERLAP; i++) {
            // copy weight
            memcpy(w_in[i].data() + config_list_all[layer_cnt][0][30]*WIDE_BUS_WIDTH/ORG_DATA_WIDTH,
                   weights[layer_cnt],
                   config_list_all[layer_cnt][0][0] * config_list_all[layer_cnt][0][0] \
                   * config_list_all[layer_cnt][0][3] \
                   * config_list_all[layer_cnt][0][7] \
                   * sizeof(DATA_T));
            // copy bias
            memcpy(w_in[i].data() + OUTPUT_LAYER_NUM * 1024 * 1024 + layer_cnt * 1024 * sizeof(BIAS_DT),
                   bias[layer_cnt],
                   config_list_all[layer_cnt][0][7] * sizeof(BIAS_DT));
        }
    }
#endif // SOC

#ifdef DEBUG_LIB
    printf("transfer weight to global memory\n");
#endif
    for(int i=0; i<OVERLAP; i++) {
        q[i]->enqueueMigrateMemObjects({*(buffer_weights[i])}, 0);
        
        //debug
        /* 
        q[i]->finish();
        q[i]->enqueueMigrateMemObjects({*(buffer_weights[i])},
                                       CL_MIGRATE_MEM_OBJECT_HOST);
        q[i]->finish();

        BIAS_DT bias_tmp[32];
        memcpy(bias_tmp,
               w_in[0].data() + OUTPUT_LAYER_NUM * 1024 * 1024 ,
               32 * sizeof(BIAS_DT));
        for(int i=0; i<32; i++) {
            printf("10 bias_in[%d]=%d\n", i, bias_tmp[i]);
        }
        */
        //debug
    }
    return 0;
}
