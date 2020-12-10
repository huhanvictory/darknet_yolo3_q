#include <stdio.h>
#include <time.h>
#include <assert.h>
#include "network.h"
#include "image.h"
#include "data.h"
#include "utils.h"
#include "blas.h"
#include "config_16x16_q.h"
#include "__merlin_define.h"
#include "crop_layer.h"
#include "connected_layer.h"
#include "gru_layer.h"
#include "rnn_layer.h"
#include "crnn_layer.h"
#include "local_layer.h"
#include "convolutional_layer.h"
#include "activation_layer.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "yolo_layer.h"
#include "normalization_layer.h"
#include "batchnorm_layer.h"
#include "maxpool_layer.h"
#include "reorg_layer.h"
#include "avgpool_layer.h"
#include "cost_layer.h"
#include "softmax_layer.h"
#include "dropout_layer.h"
#include "route_layer.h"
#include "upsample_layer.h"
#include "shortcut_layer.h"
#include "parser.h"
#include "data.h"
#define MAX_LINE 100
#define GEN_BIN_WEIGHTS (0)

void load_weights_FPGA(network* net){
    #if FPGA == 1
    __merlin_init("kernel_top.xclbin");
    //===========================================//
    //get bias data
    //===========================================//
    int32_t bias_in[OUTPUT_LAYER_NUM][1024];
    int l_cnt = 0;
    for (l_cnt = 0; l_cnt < OUTPUT_LAYER_NUM; l_cnt++)
    {
        net->index = index_conv[l_cnt];
        get_bias_data(net->layers[index_conv[l_cnt]], bias_in[l_cnt]);
#ifdef DEBUG_SIM
        int i;
        for (i = 0; i < net->layers[index_conv[l_cnt]].n; i++)
        {
            bias_in[l_cnt][i] = 0;
        }
#endif
    }
#if DEBUG_FPGA == 1
    printf("collect bias finished\n");
#endif

    //===========================================//
    //get convolution weight
    //the 1st layer, only channel depth is 3, need extend to 16 to match parallel channel computation
    //the other layers, only do data transformat
    //===========================================//
    int m, p, q, r;
    DATA_T *weights_in[OUTPUT_LAYER_NUM];
    int weight_size = 0;
    int config_format[5]; // this is for data transformat purpose
    //layer_0_weights special process, 3 channels->16 channels
    {
        weight_size = config_list_all[0][0][0] * config_list_all[0][0][0] * PARALLEL_FILTER * config_list_all[0][0][7];
        DATA_T *weights_layer_0 = (DATA_T *)malloc(weight_size * sizeof(DATA_T));
        int l0_n = net->layers[0].n;
        int l0_c = net->layers[0].c;
        int l0_size = net->layers[0].size;
        for (m = 0; m < l0_n; m++)
        {
            for (p = 0; p < PARALLEL_FILTER; p++)
            {
                for (q = 0; q < l0_size; q++)
                {
                    for (r = 0; r < l0_size; r++)
                    {
                        int index_out = m * PARALLEL_FILTER * l0_size * l0_size + p * l0_size * l0_size + q * l0_size + r;
                        int index_in = m * l0_c * l0_size * l0_size + p * l0_size * l0_size + q * l0_size + r;
                        if (p < l0_c)
                        {
                            weights_layer_0[index_out] = net->layers[0].weights[index_in];
                        }
                        else
                        {
                            weights_layer_0[index_out] = 0;
                        }
#ifdef DEBUG_SIM
                        weights_layer_0[index_out] = m + 1;
#endif
                    }
                }
            }
        }
        config_format[0] = config_list_all[0][0][0] * config_list_all[0][0][0];
        config_format[1] = PARALLEL_FILTER;
        config_format[2] = config_list_all[0][0][7];
        config_format[3] = config_list_all[0][0][7];
        config_format[4] = PARALLEL_FILTER;
        weights_in[0] = (DATA_T *)malloc(weight_size * sizeof(DATA_T));
        data_format_transform(weights_layer_0, weights_in[0], config_format);
    }
#if DEBUG_FPGA == 1
    printf("transform weights[0] finished\n");
#endif

    //copy other layer's weight
    for (l_cnt = 1; l_cnt < OUTPUT_LAYER_NUM; l_cnt++)
    {
        net->index = index_conv[l_cnt];
        config_format[0] = config_list_all[l_cnt][0][0] * config_list_all[l_cnt][0][0];
        config_format[1] = config_list_all[l_cnt][0][3];
        config_format[2] = config_list_all[l_cnt][0][7];
        config_format[3] = config_list_all[l_cnt][0][7];
        config_format[4] = PARALLEL_FILTER;
        weight_size = config_format[0] * config_format[1] * config_format[2];
        weights_in[l_cnt] = (DATA_T *)malloc(weight_size * sizeof(DATA_T));
#ifdef DEBUG_SIM
        int l_c = config_list_all[l_cnt][0][3];
        int l_n = config_list_all[l_cnt][0][7];
        int l_size = config_list_all[l_cnt][0][0];
        for (m = 0; m < l_n; m++)
        {
            for (p = 0; p < l_c; p++)
            {
                for (q = 0; q < l_size; q++)
                {
                    for (r = 0; r < l_size; r++)
                    {
                        net.layers[net.index].weights[m * l_c * l_size * l_size + p * l_size * l_size + q * l_size + r] = m + 1;
                    }
                }
            }
        }
#endif
        data_format_transform(net->layers[net->index].weights, weights_in[l_cnt], config_format);
    }
#if DEBUG_FPGA == 1
    printf("transform weights finished\n");
#endif
    __merlin_load_weight(weights_in, bias_in);

#endif
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


void get_bias_data(layer l, int32_t * bias_array) {
    int i = 0;
    for(i = 0; i < l.n; i++) {
        bias_array[i] = l.biases[i];
    }
    for(i = l.n; i < (l.n + 15)/16*16; i++) {
        bias_array[i] = 0;
    }
}

void read_data_file(int layer, DATA_T * value) {
    char file_name[] = "ir_data_quan/input_layerXXX.dat";
    file_name[27-3] = layer/100 + '0';
    file_name[28-3] = (layer%100)/10 + '0';
    file_name[29-3] = (layer%100)%10 + '0';
    char buf[MAX_LINE];  /*缓冲区*/
    FILE *fp;            /*文件指针*/
    int len;             /*行字符个数*/
//    char* path = "./";
//    char final_file_name[1000];
//    strcat(final_file_name, path);
//    strcat(final_file_name, file_name);
    printf("Input data name = %s\n", file_name);
    if((fp = fopen(file_name, "r")) == NULL) {
        perror("fail to read");
        exit (1) ;
    }
    int i = 0;
    while(fgets(buf,MAX_LINE,fp) != NULL) {
        len = strlen(buf);
        buf[len-1] = '\0';  /*去掉换行符*/
        value[i]=atof(buf);
        i++;
    }
    fclose(fp);
}

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
void write_data_file_int8_input(int layer, int8_t *value, int size) {
    char file_name[] = "input_layerXXX.dat";
    file_name[11] = layer/100 + '0';
    file_name[12] = (layer%100)/10 + '0';
    file_name[13] = (layer%100)%10 + '0';
    printf("Input data name = %s\n", file_name);
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
void write_data_file_int8(int layer, int8_t *value, int size) {
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

load_args get_base_args(network *net)
{
    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.size = net->w;

    args.min = net->min_crop;
    args.max = net->max_crop;
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.center = net->center;
    args.saturation = net->saturation;
    args.hue = net->hue;
    return args;
}

network *load_network(char *cfg, char *weights, int clear)
{
    network *net = parse_network_cfg(cfg);
    if(weights && weights[0] != 0){
        load_weights(net, weights);
    }
    if(clear) (*net->seen) = 0;
    return net;
}

size_t get_current_batch(network *net)
{
    size_t batch_num = (*net->seen)/(net->batch*net->subdivisions);
    return batch_num;
}

void reset_network_state(network *net, int b)
{
    int i;
    for (i = 0; i < net->n; ++i) {
        #ifdef GPU
        layer l = net->layers[i];
        if(l.state_gpu){
            fill_gpu(l.outputs, 0, l.state_gpu + l.outputs*b, 1);
        }
        if(l.h_gpu){
            fill_gpu(l.outputs, 0, l.h_gpu + l.outputs*b, 1);
        }
        #endif
    }
}

void reset_rnn(network *net)
{
    reset_network_state(net, 0);
}

float get_current_rate(network *net)
{
    size_t batch_num = get_current_batch(net);
    int i;
    float rate;
    if (batch_num < net->burn_in) return net->learning_rate * pow((float)batch_num / net->burn_in, net->power);
    switch (net->policy) {
        case CONSTANT:
            return net->learning_rate;
        case STEP:
            return net->learning_rate * pow(net->scale, batch_num/net->step);
        case STEPS:
            rate = net->learning_rate;
            for(i = 0; i < net->num_steps; ++i){
                if(net->steps[i] > batch_num) return rate;
                rate *= net->scales[i];
            }
            return rate;
        case EXP:
            return net->learning_rate * pow(net->gamma, batch_num);
        case POLY:
            return net->learning_rate * pow(1 - (float)batch_num / net->max_batches, net->power);
        case RANDOM:
            return net->learning_rate * pow(rand_uniform(0,1), net->power);
        case SIG:
            return net->learning_rate * (1./(1.+exp(net->gamma*(batch_num - net->step))));
        default:
            fprintf(stderr, "Policy is weird!\n");
            return net->learning_rate;
    }
}

char *get_layer_string(LAYER_TYPE a)
{
    switch(a){
        case CONVOLUTIONAL:
            return "convolutional";
        case ACTIVE:
            return "activation";
        case LOCAL:
            return "local";
        case DECONVOLUTIONAL:
            return "deconvolutional";
        case CONNECTED:
            return "connected";
        case RNN:
            return "rnn";
        case GRU:
            return "gru";
        case LSTM:
	    return "lstm";
        case CRNN:
            return "crnn";
        case MAXPOOL:
            return "maxpool";
        case REORG:
            return "reorg";
        case AVGPOOL:
            return "avgpool";
        case SOFTMAX:
            return "softmax";
        case DETECTION:
            return "detection";
        case REGION:
            return "region";
        case YOLO:
            return "yolo";
        case DROPOUT:
            return "dropout";
        case CROP:
            return "crop";
        case COST:
            return "cost";
        case ROUTE:
            return "route";
        case SHORTCUT:
            return "shortcut";
        case NORMALIZATION:
            return "normalization";
        case BATCHNORM:
            return "batchnorm";
        default:
            break;
    }
    return "none";
}

network *make_network(int n)
{
    network *net = calloc(1, sizeof(network));
    net->n = n;
    net->layers = calloc(net->n, sizeof(layer));
    net->seen = calloc(1, sizeof(size_t));
    net->t    = calloc(1, sizeof(int));
    net->cost = calloc(1, sizeof(float));
    return net;
}

void forward_network(network *netp)
{
#ifdef GPU
    if(netp->gpu_index >= 0){
        forward_network_gpu(netp);   
        return;
    }
#endif
    network net = *netp;
    int i;
    for(i = 0; i < net.n; ++i){
        net.index = i;
        layer l = net.layers[i];
        if(l.delta){
            fill_cpu(l.outputs * l.batch, 0, l.delta, 1);
        }
        #if OUTPUT_REF == 1
        if(i != 82 && i != 94 && i != 106
                && i != 83 && i != 86 && i!= 95 && i!= 98) {
            write_data_file_int8_input(i, net.input, l.inputs);
        }
        #endif
        double time;
        time=what_time_is_it_now();
        l.forward(l, net);
        printf("layer %d in %f seconds.\n", i, what_time_is_it_now()-time); 
        #if OUTPUT_REF == 1
        if(i != 82 && i != 94 && i != 106) {
            write_data_file_int8(i, l.output, l.outputs);
        }
        #endif
        net.input = l.output;
        if(l.truth) {
            net.truth = l.output;
        }
    }
    calc_network_cost(netp);
}

void forward_network_fpga(network *netp, int * test_cfg)
{
    int batch = 1;
    int layer_min = test_cfg[0];
    int layer_max = test_cfg[1];
    int debug_layer = test_cfg[2];
    //int cfg_channel = test_cfg[2];
    int cfg_filter = test_cfg[3];
    #if DEBUG_FPGA == 1
    printf("testing cfg = %d %d %d %d\n", layer_min, layer_max, debug_layer, cfg_filter);
    #endif
    network net = *netp;
    DATA_T * layer_0_in = malloc(sizeof(DATA_T)*416*416*3);
    float* yolo1_out = calloc(12675*batch, sizeof(float)); 
    float* yolo2_out = calloc(50700*batch, sizeof(float)); 
    float* yolo3_out = calloc(202800*batch, sizeof(float)); 
    
    //===========================================//
    //get bias data
    //===========================================//
    int32_t bias_in[OUTPUT_LAYER_NUM][1024];
    int l_cnt = 0;
    for(l_cnt = 0; l_cnt < OUTPUT_LAYER_NUM; l_cnt++){
        net.index = index_conv[l_cnt];
        get_bias_data(net.layers[index_conv[l_cnt]], bias_in[l_cnt]);
        #ifdef DEBUG_SIM
        int i;
        for(i = 0; i < net.layers[index_conv[l_cnt]].n; i++) {
            bias_in[l_cnt][i] = 0;
        }
        #endif
    }
 #if(GEN_BIN_WEIGHTS)
    FILE *fp = fopen("raw_bias.bin", "wb"); // this is the file containing the biases ready to be transfered to the FPGA
    for (l_cnt = 0; l_cnt < OUTPUT_LAYER_NUM; l_cnt++)
    {
        fwrite(bias_in[l_cnt], sizeof(int32_t), 1024, fp);
    }
    fclose(fp);
#endif   

    #if DEBUG_FPGA == 1
    printf("collect bias finished\n");
    #endif
    
    //===========================================//
    //get convolution weight
    //the 1st layer, only channel depth is 3, need extend to 16 to match parallel channel computation
    //the other layers, only do data transformat
    //===========================================//
    int m,p,q,r;
    DATA_T *weights_in[OUTPUT_LAYER_NUM];
    int weight_size = 0;
    int config_format[5]; // this is for data transformat purpose
    //layer_0_weights special process, 3 channels->16 channels
    {
        weight_size = config_list_all[0][0][0] * config_list_all[0][0][0] * PARALLEL_FILTER * config_list_all[0][0][7];
        DATA_T *weights_layer_0 = (DATA_T*)malloc(weight_size * sizeof(DATA_T));
        int l0_n = net.layers[0].n;
        int l0_c = net.layers[0].c;
        int l0_size = net.layers[0].size;
        for(m = 0; m < l0_n; m++){
            for(p = 0; p < PARALLEL_FILTER; p++){
                for(q = 0; q < l0_size; q++){
                    for(r = 0; r < l0_size; r++){
                        int index_out = m*PARALLEL_FILTER*l0_size*l0_size + p*l0_size*l0_size + q*l0_size + r;
                        int index_in = m*l0_c*l0_size*l0_size + p*l0_size*l0_size + q*l0_size + r;
                        if(p < l0_c) {
                            weights_layer_0[index_out] = net.layers[0].weights[index_in];
                        } else {
                            weights_layer_0[index_out] = 0;
                        }
                        #ifdef DEBUG_SIM
                        weights_layer_0[index_out] = m+1;
                        #endif
                    }
                }
            }
        }
        config_format[0] = config_list_all[0][0][0]*config_list_all[0][0][0];
        config_format[1] = PARALLEL_FILTER;
        config_format[2] = config_list_all[0][0][7];
        config_format[3] = config_list_all[0][0][7];
        config_format[4] = PARALLEL_FILTER;
        weights_in[0] = (DATA_T*)malloc(weight_size * sizeof(DATA_T));
        data_format_transform(weights_layer_0, weights_in[0], config_format);
    }

 #if(GEN_BIN_WEIGHTS)
    fp = fopen("raw_wgts.bin", "wb");
    for (l_cnt = 0; l_cnt < OUTPUT_LAYER_NUM; ++l_cnt)
    {
        int weight_len = -1;
        if (l_cnt == 0)
        {
            weight_len = config_list_all[0][0][0] * config_list_all[0][0][0] * PARALLEL_FILTER * config_list_all[0][0][7];
        }
        else
        {
            weight_len = config_list_all[l_cnt][0][0] * config_list_all[l_cnt][0][0] * config_list_all[l_cnt][0][3] * config_list_all[l_cnt][0][7];
            ;
        }
        fwrite(weights_in[l_cnt], sizeof(DATA_T), weight_len, fp);
    }
    fclose(fp);
#endif   
    #if DEBUG_FPGA == 1
    printf("transform weights[0] finished\n");
    #endif
    
    //copy other layer's weight
    for(l_cnt = 1; l_cnt < OUTPUT_LAYER_NUM; l_cnt++){
        net.index = index_conv[l_cnt];
        config_format[0] = config_list_all[l_cnt][0][0] * config_list_all[l_cnt][0][0];
        config_format[1] = config_list_all[l_cnt][0][3];
        config_format[2] = config_list_all[l_cnt][0][7];
        config_format[3] = config_list_all[l_cnt][0][7];
        config_format[4] = PARALLEL_FILTER;
        weight_size = config_format[0]* config_format[1] *config_format[2];
        weights_in[l_cnt] = (DATA_T*)malloc(weight_size * sizeof(DATA_T));
        #ifdef DEBUG_SIM
        int l_c = config_list_all[l_cnt][0][3];
        int l_n = config_list_all[l_cnt][0][7];
        int l_size = config_list_all[l_cnt][0][0];
        for(m = 0; m < l_n; m++){
            for(p = 0; p < l_c; p++){
                for(q = 0; q < l_size; q++){
                    for(r = 0; r < l_size; r++){
                        net.layers[net.index].weights[m*l_c*l_size*l_size + p*l_size*l_size + q*l_size + r] = m+1;
                    }
                }
            }
        }
        #endif
        data_format_transform(net.layers[net.index].weights, weights_in[l_cnt], config_format);
        #if DEBUG_FPGA == 1
        printf("transform weights[%d] finished\n", l_cnt);
        #endif
    }
    #if DEBUG_FPGA == 1
    printf("transform weights finished\n");
    #endif

    //layer 0 input data
    //TODO: in batch mode, it need to add offset for input images
    net.index = 0;
    layer l0 = net.layers[0];
    //for(p = 0; p < batch; p++){
        for(m = 0; m < l0.inputs; m++){
            layer_0_in[m] = net.input[m];
        }
    //}
    //write_data_file(300, layer_0_in, l0.inputs);//debug layer  
    //write_data_file_int8_input(998, layer_0_in, sizeof(DATA_T)*416*416*3);//debug layer  
    
    //===========================================//
    //loading weight and bias to global memory, only need once for different images
    //===========================================//
    #if DEBUG_FPGA == 1
    printf("loading weight\n");
    #endif
    #if FPGA == 1
    __merlin_load_weight(weights_in, bias_in);
    #endif
    int debug_config[10];
    #if DEBUG_CPU == 1
    debug_config[0] = debug_layer;//new layer
    if(layer_min == 58)
        debug_config[1] = 81;// old layer
    else if(layer_min == 59)
        debug_config[1] = 84;// old layer
    else if(layer_min == 66)
        debug_config[1] = 93;// old layer
    else if(layer_min == 67)
        debug_config[1] = 96;// old layer
    else if(layer_min == 74)
        debug_config[1] = 105;// old layer
    else
        debug_config[1] = index_conv[layer_min + 1] - 1;// old layer
    debug_config[2] = layer_min;
    debug_config[3] = layer_max;
    debug_config[4] = config_list_all[layer_min][0][5];
    net.index = debug_config[1];
    debug_config[5] = net.layers[debug_config[1]].c;

    int old_layer_x = debug_config[1];
    int new_layer_x = debug_config[2];
    int i_w = config_list_all[new_layer_x][0][1];
    int i_h = config_list_all[new_layer_x][0][2];
    int i_c = config_list_all[new_layer_x][0][3];
    int data_size = i_w * i_h * i_c;
    #if DEBUG_FPGA == 1
    printf("debug_layer:%d, old layer:%d, data_size:%d\n", new_layer_x, old_layer_x, data_size);
    #endif

    DATA_T * layer_x_in = malloc(sizeof(DATA_T)*data_size);
    read_data_file(old_layer_x, layer_x_in);
    //write_data_file(301, layer_x_in, l0.inputs);//debug layer  
    #ifdef DEBUG_SIM
    int l_c = net.layers[old_layer_x].c;
    int l_n = net.layers[old_layer_x].n;
    int l_h = net.layers[old_layer_x].h;
    int l_w = net.layers[old_layer_x].w;
    int l_size = net.layers[old_layer_x].size;

    for(p = 0; p < l_c; p++){
        for(q = 0; q < l_h; q++){
            for(r = 0; r < l_w; r++){
                layer_x_in[p*l_h*l_w + q*l_w + r] = q%104+1;
            }
        }
    }
    #endif // DEBUG_SIM
    #endif
    
    //===========================================//
    //fpga acceleration
    //currently including conv, shorcut, route, upsample layers
    //===========================================//
    #if DEBUG_FPGA == 1
    printf("detecting\n");
    #endif
    struct timeval tv_start, tv_end;
    double exe_time;
    gettimeofday(&tv_start, NULL);
//    write_data_file(302, layer_0_in, l0.inputs);//debug layer  
    #if FPGA == 1
    #if DEBUG_CPU == 1
    __merlin_exec_top_kernel_overlap(layer_x_in, yolo1_out, yolo2_out, yolo3_out, 1, debug_config);
    #else
    
    debug_config[0] = 0;
    debug_config[1] = 0;
    debug_config[2] = 0;
    debug_config[3] = 74;
    debug_config[4] = 416;
    debug_config[5] = 3;

    batch = 4;
    DATA_T * batched_layer_0_in = malloc(sizeof(DATA_T)*416*416*3*batch);
    for(p = 0; p < batch; p++) {
        memcpy(batched_layer_0_in + (416 * 416 * 3) * p, layer_0_in, 416 * 416 * 3 * sizeof(DATA_T));
    }
    float* batched_yolo1_out = calloc(12675*batch, sizeof(float)); 
    float* batched_yolo2_out = calloc(50700*batch, sizeof(float)); 
    float* batched_yolo3_out = calloc(202800*batch, sizeof(float));
    
    //FILE *pfp = fopen("inp_img.bin", "ab");
    //fwrite(layer_0_in, sizeof(DATA_T), (416 * 416 *3), pfp);
    FILE *pfp = fopen("inp_img.bin", "rb");
    fread(batched_layer_0_in, sizeof(DATA_T), (416 * 416 *3) * 4, pfp);

    //__merlin_exec_top_kernel_overlap(layer_0_in, yolo1_out, yolo2_out, yolo3_out, batch, debug_config);
    __merlin_exec_top_kernel_overlap(batched_layer_0_in, batched_yolo1_out, batched_yolo2_out, batched_yolo3_out, batch, debug_config);
    memcpy(yolo1_out, batched_yolo1_out + 12675 * 3, 12675 * sizeof(float));
    memcpy(yolo2_out, batched_yolo2_out + 50700 * 3, 50700 * sizeof(float));
    memcpy(yolo3_out, batched_yolo3_out + 202800 * 3, 202800 * sizeof(float));
    
    #endif // DEBUG_CPU
    #endif
    #if DEBUG_FPGA == 1
    printf("finish opencl kernel\n");
    #endif
    gettimeofday(&tv_end, NULL);
    exe_time = (tv_end.tv_sec - tv_start.tv_sec) * 1000.0 + (tv_end.tv_usec - tv_start.tv_usec)/1000.0;                
    printf("E2E time %f ms \n",exe_time);

    //===========================================//
    //get yolo data to original data structure
    //===========================================//
    int i_q;
    { //yolo1
        layer l = net.layers[82];
        for(i_q = 0; i_q < l.outputs;  ++ i_q) 
            l.yolo_out[i_q] = yolo1_out[i_q];
    }
    { //yolo2
        layer l = net.layers[94];
        for(i_q = 0; i_q < l.outputs;  ++ i_q) 
            l.yolo_out[i_q] = yolo2_out[i_q];
    }
    { //yolo3
        layer l = net.layers[106];
        for(i_q = 0; i_q < l.outputs;  ++ i_q) 
            l.yolo_out[i_q] = yolo3_out[i_q];
    }
    calc_network_cost(netp);
}

void update_network(network *netp)
{
#ifdef GPU
    if(netp->gpu_index >= 0){
        update_network_gpu(netp);   
        return;
    }
#endif
    network net = *netp;
    int i;
    update_args a = {0};
    a.batch = net.batch*net.subdivisions;
    a.learning_rate = get_current_rate(netp);
    a.momentum = net.momentum;
    a.decay = net.decay;
    a.adam = net.adam;
    a.B1 = net.B1;
    a.B2 = net.B2;
    a.eps = net.eps;
    ++*net.t;
    a.t = *net.t;

    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.update){
            l.update(l, a);
        }
    }
}

void calc_network_cost(network *netp)
{
    network net = *netp;
    int i;
    float sum = 0;
    int count = 0;
    for(i = 0; i < net.n; ++i){
        if(net.layers[i].cost){
            sum += net.layers[i].cost[0];
            ++count;
        }
    }
    *net.cost = sum/count;
}

int get_predicted_class_network(network *net)
{
    return max_index(net->output, net->outputs);
}

void backward_network(network *netp)
{
#ifdef GPU
    if(netp->gpu_index >= 0){
        backward_network_gpu(netp);   
        return;
    }
#endif
    network net = *netp;
    int i;
    network orig = net;
    for(i = net.n-1; i >= 0; --i){
        layer l = net.layers[i];
        if(l.stopbackward) break;
        if(i == 0){
            net = orig;
        }else{
            layer prev = net.layers[i-1];
            net.input = prev.output;
            net.delta = prev.delta;
        }
        net.index = i;
        l.backward(l, net);
    }
}

float train_network_datum(network *net)
{
    *net->seen += net->batch;
    net->train = 1;
    forward_network(net);
    backward_network(net);
    float error = *net->cost;
    if(((*net->seen)/net->batch)%net->subdivisions == 0) update_network(net);
    return error;
}

float train_network_sgd(network *net, data d, int n)
{
    int batch = net->batch;

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        get_random_batch(d, batch, net->input, net->truth);
        float err = train_network_datum(net);
        sum += err;
    }
    return (float)sum/(n*batch);
}

float train_network(network *net, data d)
{
    assert(d.X.rows % net->batch == 0);
    int batch = net->batch;
    int n = d.X.rows / batch;

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        get_next_batch(d, batch, i*batch, net->input, net->truth);
        float err = train_network_datum(net);
        sum += err;
    }
    return (float)sum/(n*batch);
}

void set_temp_network(network *net, float t)
{
    int i;
    for(i = 0; i < net->n; ++i){
        net->layers[i].temperature = t;
    }
}


void set_batch_network(network *net, int b)
{
    net->batch = b;
    int i;
    for(i = 0; i < net->n; ++i){
        net->layers[i].batch = b;
#ifdef CUDNN
        if(net->layers[i].type == CONVOLUTIONAL){
            cudnn_convolutional_setup(net->layers + i);
        }
        if(net->layers[i].type == DECONVOLUTIONAL){
            layer *l = net->layers + i;
            cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, l->out_h, l->out_w);
            cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1); 
        }
#endif
    }
}

int resize_network(network *net, int w, int h)
{
#ifdef GPU
    cuda_set_device(net->gpu_index);
    cuda_free(net->workspace);
#endif
    int i;
    //if(w == net->w && h == net->h) return 0;
    net->w = w;
    net->h = h;
    int inputs = 0;
    size_t workspace_size = 0;
    //fprintf(stderr, "Resizing to %d x %d...\n", w, h);
    //fflush(stderr);
    for (i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            resize_convolutional_layer(&l, w, h);
        }else if(l.type == CROP){
            resize_crop_layer(&l, w, h);
        }else if(l.type == MAXPOOL){
            resize_maxpool_layer(&l, w, h);
        }else if(l.type == REGION){
            resize_region_layer(&l, w, h);
        }else if(l.type == YOLO){
            resize_yolo_layer(&l, w, h);
        }else if(l.type == ROUTE){
            resize_route_layer(&l, net);
        }else if(l.type == SHORTCUT){
            resize_shortcut_layer(&l, w, h);
        }else if(l.type == UPSAMPLE){
            resize_upsample_layer(&l, w, h);
        }else if(l.type == REORG){
            resize_reorg_layer(&l, w, h);
        }else if(l.type == AVGPOOL){
            resize_avgpool_layer(&l, w, h);
        }else if(l.type == NORMALIZATION){
            resize_normalization_layer(&l, w, h);
        }else if(l.type == COST){
            resize_cost_layer(&l, inputs);
        }else{
            error("Cannot resize this type of layer");
        }
        if(l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        if(l.workspace_size > 2000000000) assert(0);
        inputs = l.outputs;
        net->layers[i] = l;
        w = l.out_w;
        h = l.out_h;
        if(l.type == AVGPOOL) break;
    }
    layer out = get_network_output_layer(net);
    net->inputs = net->layers[0].inputs;
    net->outputs = out.outputs;
    net->truths = out.outputs;
    if(net->layers[net->n-1].truths) net->truths = net->layers[net->n-1].truths;
    net->output = out.output;
    free(net->input);
    free(net->truth);
    net->input = calloc(net->inputs*net->batch, sizeof(float));
    net->truth = calloc(net->truths*net->batch, sizeof(float));
#ifdef GPU
    if(gpu_index >= 0){
        cuda_free(net->input_gpu);
        cuda_free(net->truth_gpu);
        net->input_gpu = cuda_make_array(net->input, net->inputs*net->batch);
        net->truth_gpu = cuda_make_array(net->truth, net->truths*net->batch);
        if(workspace_size){
            net->workspace = cuda_make_array(0, (workspace_size-1)/sizeof(float)+1);
        }
    }else {
        free(net->workspace);
        net->workspace = calloc(1, workspace_size);
    }
#else
    free(net->workspace);
    net->workspace = calloc(1, workspace_size);
#endif
    //fprintf(stderr, " Done!\n");
    return 0;
}

layer get_network_detection_layer(network *net)
{
    int i;
    for(i = 0; i < net->n; ++i){
        if(net->layers[i].type == DETECTION){
            return net->layers[i];
        }
    }
    fprintf(stderr, "Detection layer not found!!\n");
    layer l = {0};
    return l;
}

image get_network_image_layer(network *net, int i)
{
    layer l = net->layers[i];
#ifdef GPU
    //cuda_pull_array(l.output_gpu, l.output, l.outputs);
#endif
    if (l.out_w && l.out_h && l.out_c){
        return float_to_image(l.out_w, l.out_h, l.out_c, l.output);
    }
    image def = {0};
    return def;
}

image get_network_image(network *net)
{
    int i;
    for(i = net->n-1; i >= 0; --i){
        image m = get_network_image_layer(net, i);
        if(m.h != 0) return m;
    }
    image def = {0};
    return def;
}

void visualize_network(network *net)
{
    image *prev = 0;
    int i;
    char buff[256];
    for(i = 0; i < net->n; ++i){
        sprintf(buff, "Layer %d", i);
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            prev = visualize_convolutional_layer(l, buff, prev);
        }
    } 
}

void top_predictions(network *net, int k, int *index)
{
    top_k(net->output, net->outputs, k, index);
}


float *network_predict_fpga(network *net, float *input, int * test_cfg)
{
    #if FPGA == 1
    __merlin_init("kernel_top.xclbin");
    #endif
    network orig = *net;
    // net->input = input;
    int i_q;
    #if DEBUG_FPGA == 1
    double time;
    time=what_time_is_it_now();
    #endif
//    write_data_file(1, net->input , net->inputs);//debug layer  
    for (i_q = 0; i_q < net->inputs; ++i_q)
    {
        net->input[i_q] = xilinx_quantizer_shift(round(input[i_q] * 64), 0);
        //net->input[i_q] = round(input[i_q] * 64);
    }
//    write_data_file(0, net->input , net->inputs);//debug layer  
    #if DEBUG_FPGA == 1
    printf("first layer quantize in %f seconds.\n", what_time_is_it_now()-time); 
    #endif
    int sum_aq = sum_f(net->input, net->inputs);
    
    net->truth = 0;
    net->train = 0;
    net->delta = 0;
    forward_network_fpga(net, test_cfg);
    float *out = net->output;
    *net = orig;
    #if FPGA == 1
    __merlin_release();
    #endif
    return out;
}
float *network_predict(network *net, float *input)
{
    network orig = *net;
    // net->input = input;
    int i_q;
    //write_data_file_int8_input(1, net->input , net->inputs);//debug layer  
    for (i_q = 0; i_q < net->inputs; ++i_q)
    {
        net->input[i_q] = xilinx_quantizer_shift(round(input[i_q] * 64), 0);
        //net->input[i_q] = round(input[i_q] * 64);
    }
    //write_data_file_int8_input(0, net->input , net->inputs);//debug layer  
    int sum_aq = sum_f(net->input, net->inputs);
    
    net->truth = 0;
    net->train = 0;
    net->delta = 0;
    forward_network(net);
    float *out = net->output;
    *net = orig;
    return out;
}

int num_detections(network *net, float thresh)
{
    int i;
    int s = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO){
            s += yolo_num_detections(l, thresh);
        }
        if(l.type == DETECTION || l.type == REGION){
            s += l.w*l.h*l.n;
        }
    }
    return s;
}

detection *make_network_boxes(network *net, float thresh, int *num)
{
    layer l = net->layers[net->n - 1];
    int i;
    int nboxes = num_detections(net, thresh);
    if(num) *num = nboxes;
    detection *dets = calloc(nboxes, sizeof(detection));
    for(i = 0; i < nboxes; ++i){
        dets[i].prob = calloc(l.classes, sizeof(float));
        if(l.coords > 4){
            dets[i].mask = calloc(l.coords-4, sizeof(float));
        }
    }
    return dets;
}

void fill_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, detection *dets)
{
    int j;
    for(j = 0; j < net->n; ++j){
        layer l = net->layers[j];
        if(l.type == YOLO){
            int count = get_yolo_detections(l, w, h, net->w, net->h, thresh, map, relative, dets);
            dets += count;
        }
        if(l.type == REGION){
            get_region_detections(l, w, h, net->w, net->h, thresh, map, hier, relative, dets);
            dets += l.w*l.h*l.n;
        }
        if(l.type == DETECTION){
            get_detection_detections(l, w, h, thresh, dets);
            dets += l.w*l.h*l.n;
        }
    }
}

detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num)
{
    detection *dets = make_network_boxes(net, thresh, num);
    fill_network_boxes(net, w, h, thresh, hier, map, relative, dets);
    return dets;
}

void free_detections(detection *dets, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        free(dets[i].prob);
        if(dets[i].mask) free(dets[i].mask);
    }
    free(dets);
}

float *network_predict_image(network *net, image im)
{
    image imr = letterbox_image(im, net->w, net->h);
    set_batch_network(net, 1);
    float *p = network_predict(net, imr.data);
    free_image(imr);
    return p;
}

int network_width(network *net){return net->w;}
int network_height(network *net){return net->h;}

matrix network_predict_data_multi(network *net, data test, int n)
{
    int i,j,b,m;
    int k = net->outputs;
    matrix pred = make_matrix(test.X.rows, k);
    float *X = calloc(net->batch*test.X.rows, sizeof(float));
    for(i = 0; i < test.X.rows; i += net->batch){
        for(b = 0; b < net->batch; ++b){
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
        }
        for(m = 0; m < n; ++m){
            float *out = network_predict(net, X);
            for(b = 0; b < net->batch; ++b){
                if(i+b == test.X.rows) break;
                for(j = 0; j < k; ++j){
                    pred.vals[i+b][j] += out[j+b*k]/n;
                }
            }
        }
    }
    free(X);
    return pred;   
}

matrix network_predict_data(network *net, data test)
{
    int i,j,b;
    int k = net->outputs;
    matrix pred = make_matrix(test.X.rows, k);
    float *X = calloc(net->batch*test.X.cols, sizeof(float));
    for(i = 0; i < test.X.rows; i += net->batch){
        for(b = 0; b < net->batch; ++b){
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
        }
        float *out = network_predict(net, X);
        for(b = 0; b < net->batch; ++b){
            if(i+b == test.X.rows) break;
            for(j = 0; j < k; ++j){
                pred.vals[i+b][j] = out[j+b*k];
            }
        }
    }
    free(X);
    return pred;   
}

void print_network(network *net)
{
    int i,j;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        float *output = l.output;
        int n = l.outputs;
        float mean = mean_array(output, n);
        float vari = variance_array(output, n);
        fprintf(stderr, "Layer %d - Mean: %f, Variance: %f\n",i,mean, vari);
        if(n > 100) n = 100;
        for(j = 0; j < n; ++j) fprintf(stderr, "%f, ", output[j]);
        if(n == 100)fprintf(stderr,".....\n");
        fprintf(stderr, "\n");
    }
}

void compare_networks(network *n1, network *n2, data test)
{
    matrix g1 = network_predict_data(n1, test);
    matrix g2 = network_predict_data(n2, test);
    int i;
    int a,b,c,d;
    a = b = c = d = 0;
    for(i = 0; i < g1.rows; ++i){
        int truth = max_index(test.y.vals[i], test.y.cols);
        int p1 = max_index(g1.vals[i], g1.cols);
        int p2 = max_index(g2.vals[i], g2.cols);
        if(p1 == truth){
            if(p2 == truth) ++d;
            else ++c;
        }else{
            if(p2 == truth) ++b;
            else ++a;
        }
    }
    printf("%5d %5d\n%5d %5d\n", a, b, c, d);
    float num = pow((abs(b - c) - 1.), 2.);
    float den = b + c;
    printf("%f\n", num/den); 
}

float network_accuracy(network *net, data d)
{
    matrix guess = network_predict_data(net, d);
    float acc = matrix_topk_accuracy(d.y, guess,1);
    free_matrix(guess);
    return acc;
}

float *network_accuracies(network *net, data d, int n)
{
    static float acc[2];
    matrix guess = network_predict_data(net, d);
    acc[0] = matrix_topk_accuracy(d.y, guess, 1);
    acc[1] = matrix_topk_accuracy(d.y, guess, n);
    free_matrix(guess);
    return acc;
}

layer get_network_output_layer(network *net)
{
    int i;
    for(i = net->n - 1; i >= 0; --i){
        if(net->layers[i].type != COST) break;
    }
    return net->layers[i];
}

float network_accuracy_multi(network *net, data d, int n)
{
    matrix guess = network_predict_data_multi(net, d, n);
    float acc = matrix_topk_accuracy(d.y, guess,1);
    free_matrix(guess);
    return acc;
}

void free_network(network *net)
{
    int i;
    for(i = 0; i < net->n; ++i){
        free_layer(net->layers[i]);
    }
    free(net->layers);
    if(net->input) free(net->input);
    if(net->truth) free(net->truth);
#ifdef GPU
    if(net->input_gpu) cuda_free(net->input_gpu);
    if(net->truth_gpu) cuda_free(net->truth_gpu);
#endif
    free(net);
}

// Some day...
// ^ What the hell is this comment for?


layer network_output_layer(network *net)
{
    int i;
    for(i = net->n - 1; i >= 0; --i){
        if(net->layers[i].type != COST) break;
    }
    return net->layers[i];
}

int network_inputs(network *net)
{
    return net->layers[0].inputs;
}

int network_outputs(network *net)
{
    return network_output_layer(net).outputs;
}

float *network_output(network *net)
{
    return network_output_layer(net).output;
}

#ifdef GPU

void forward_network_gpu(network *netp)
{
    network net = *netp;
    cuda_set_device(net.gpu_index);
    cuda_push_array(net.input_gpu, net.input, net.inputs*net.batch);
    if(net.truth){
        cuda_push_array(net.truth_gpu, net.truth, net.truths*net.batch);
    }

    int i;
    for(i = 0; i < net.n; ++i){
        net.index = i;
        layer l = net.layers[i];
        if(l.delta_gpu){
            fill_gpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
        }
        l.forward_gpu(l, net);
        net.input_gpu = l.output_gpu;
        net.input = l.output;
        if(l.truth) {
            net.truth_gpu = l.output_gpu;
            net.truth = l.output;
        }
    }
    pull_network_output(netp);
    calc_network_cost(netp);
}

void backward_network_gpu(network *netp)
{
    int i;
    network net = *netp;
    network orig = net;
    cuda_set_device(net.gpu_index);
    for(i = net.n-1; i >= 0; --i){
        layer l = net.layers[i];
        if(l.stopbackward) break;
        if(i == 0){
            net = orig;
        }else{
            layer prev = net.layers[i-1];
            net.input = prev.output;
            net.delta = prev.delta;
            net.input_gpu = prev.output_gpu;
            net.delta_gpu = prev.delta_gpu;
        }
        net.index = i;
        l.backward_gpu(l, net);
    }
}

void update_network_gpu(network *netp)
{
    network net = *netp;
    cuda_set_device(net.gpu_index);
    int i;
    update_args a = {0};
    a.batch = net.batch*net.subdivisions;
    a.learning_rate = get_current_rate(netp);
    a.momentum = net.momentum;
    a.decay = net.decay;
    a.adam = net.adam;
    a.B1 = net.B1;
    a.B2 = net.B2;
    a.eps = net.eps;
    ++*net.t;
    a.t = (*net.t);

    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.update_gpu){
            l.update_gpu(l, a);
        }
    }
}

void harmless_update_network_gpu(network *netp)
{
    network net = *netp;
    cuda_set_device(net.gpu_index);
    int i;
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.weight_updates_gpu) fill_gpu(l.nweights, 0, l.weight_updates_gpu, 1);
        if(l.bias_updates_gpu) fill_gpu(l.nbiases, 0, l.bias_updates_gpu, 1);
        if(l.scale_updates_gpu) fill_gpu(l.nbiases, 0, l.scale_updates_gpu, 1);
    }
}

typedef struct {
    network *net;
    data d;
    float *err;
} train_args;

void *train_thread(void *ptr)
{
    train_args args = *(train_args*)ptr;
    free(ptr);
    cuda_set_device(args.net->gpu_index);
    *args.err = train_network(args.net, args.d);
    return 0;
}

pthread_t train_network_in_thread(network *net, data d, float *err)
{
    pthread_t thread;
    train_args *ptr = (train_args *)calloc(1, sizeof(train_args));
    ptr->net = net;
    ptr->d = d;
    ptr->err = err;
    if(pthread_create(&thread, 0, train_thread, ptr)) error("Thread creation failed");
    return thread;
}

void merge_weights(layer l, layer base)
{
    if (l.type == CONVOLUTIONAL) {
        axpy_cpu(l.n, 1, l.bias_updates, 1, base.biases, 1);
        axpy_cpu(l.nweights, 1, l.weight_updates, 1, base.weights, 1);
        if (l.scales) {
            axpy_cpu(l.n, 1, l.scale_updates, 1, base.scales, 1);
        }
    } else if(l.type == CONNECTED) {
        axpy_cpu(l.outputs, 1, l.bias_updates, 1, base.biases, 1);
        axpy_cpu(l.outputs*l.inputs, 1, l.weight_updates, 1, base.weights, 1);
    }
}

void scale_weights(layer l, float s)
{
    if (l.type == CONVOLUTIONAL) {
        scal_cpu(l.n, s, l.biases, 1);
        scal_cpu(l.nweights, s, l.weights, 1);
        if (l.scales) {
            scal_cpu(l.n, s, l.scales, 1);
        }
    } else if(l.type == CONNECTED) {
        scal_cpu(l.outputs, s, l.biases, 1);
        scal_cpu(l.outputs*l.inputs, s, l.weights, 1);
    }
}


void pull_weights(layer l)
{
    if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
        cuda_pull_array(l.biases_gpu, l.bias_updates, l.n);
        cuda_pull_array(l.weights_gpu, l.weight_updates, l.nweights);
        if(l.scales) cuda_pull_array(l.scales_gpu, l.scale_updates, l.n);
    } else if(l.type == CONNECTED){
        cuda_pull_array(l.biases_gpu, l.bias_updates, l.outputs);
        cuda_pull_array(l.weights_gpu, l.weight_updates, l.outputs*l.inputs);
    }
}

void push_weights(layer l)
{
    if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
        cuda_push_array(l.biases_gpu, l.biases, l.n);
        cuda_push_array(l.weights_gpu, l.weights, l.nweights);
        if(l.scales) cuda_push_array(l.scales_gpu, l.scales, l.n);
    } else if(l.type == CONNECTED){
        cuda_push_array(l.biases_gpu, l.biases, l.outputs);
        cuda_push_array(l.weights_gpu, l.weights, l.outputs*l.inputs);
    }
}

void distribute_weights(layer l, layer base)
{
    if (l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL) {
        cuda_push_array(l.biases_gpu, base.biases, l.n);
        cuda_push_array(l.weights_gpu, base.weights, l.nweights);
        if (base.scales) cuda_push_array(l.scales_gpu, base.scales, l.n);
    } else if (l.type == CONNECTED) {
        cuda_push_array(l.biases_gpu, base.biases, l.outputs);
        cuda_push_array(l.weights_gpu, base.weights, l.outputs*l.inputs);
    }
}


/*

   void pull_updates(layer l)
   {
   if(l.type == CONVOLUTIONAL){
   cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
   cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
   if(l.scale_updates) cuda_pull_array(l.scale_updates_gpu, l.scale_updates, l.n);
   } else if(l.type == CONNECTED){
   cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
   cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.outputs*l.inputs);
   }
   }

   void push_updates(layer l)
   {
   if(l.type == CONVOLUTIONAL){
   cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
   cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
   if(l.scale_updates) cuda_push_array(l.scale_updates_gpu, l.scale_updates, l.n);
   } else if(l.type == CONNECTED){
   cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
   cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.outputs*l.inputs);
   }
   }

   void update_layer(layer l, network net)
   {
   int update_batch = net.batch*net.subdivisions;
   float rate = get_current_rate(net);
   l.t = get_current_batch(net);
   if(l.update_gpu){
   l.update_gpu(l, update_batch, rate*l.learning_rate_scale, net.momentum, net.decay);
   }
   }
   void merge_updates(layer l, layer base)
   {
   if (l.type == CONVOLUTIONAL) {
   axpy_cpu(l.n, 1, l.bias_updates, 1, base.bias_updates, 1);
   axpy_cpu(l.nweights, 1, l.weight_updates, 1, base.weight_updates, 1);
   if (l.scale_updates) {
   axpy_cpu(l.n, 1, l.scale_updates, 1, base.scale_updates, 1);
   }
   } else if(l.type == CONNECTED) {
   axpy_cpu(l.outputs, 1, l.bias_updates, 1, base.bias_updates, 1);
   axpy_cpu(l.outputs*l.inputs, 1, l.weight_updates, 1, base.weight_updates, 1);
   }
   }

   void distribute_updates(layer l, layer base)
   {
   if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
   cuda_push_array(l.bias_updates_gpu, base.bias_updates, l.n);
   cuda_push_array(l.weight_updates_gpu, base.weight_updates, l.nweights);
   if(base.scale_updates) cuda_push_array(l.scale_updates_gpu, base.scale_updates, l.n);
   } else if(l.type == CONNECTED){
   cuda_push_array(l.bias_updates_gpu, base.bias_updates, l.outputs);
   cuda_push_array(l.weight_updates_gpu, base.weight_updates, l.outputs*l.inputs);
   }
   }
 */

/*
   void sync_layer(network *nets, int n, int j)
   {
   int i;
   network net = nets[0];
   layer base = net.layers[j];
   scale_weights(base, 0);
   for (i = 0; i < n; ++i) {
   cuda_set_device(nets[i].gpu_index);
   layer l = nets[i].layers[j];
   pull_weights(l);
   merge_weights(l, base);
   }
   scale_weights(base, 1./n);
   for (i = 0; i < n; ++i) {
   cuda_set_device(nets[i].gpu_index);
   layer l = nets[i].layers[j];
   distribute_weights(l, base);
   }
   }
 */

void sync_layer(network **nets, int n, int j)
{
    int i;
    network *net = nets[0];
    layer base = net->layers[j];
    scale_weights(base, 0);
    for (i = 0; i < n; ++i) {
        cuda_set_device(nets[i]->gpu_index);
        layer l = nets[i]->layers[j];
        pull_weights(l);
        merge_weights(l, base);
    }
    scale_weights(base, 1./n);
    for (i = 0; i < n; ++i) {
        cuda_set_device(nets[i]->gpu_index);
        layer l = nets[i]->layers[j];
        distribute_weights(l, base);
    }
}

typedef struct{
    network **nets;
    int n;
    int j;
} sync_args;

void *sync_layer_thread(void *ptr)
{
    sync_args args = *(sync_args*)ptr;
    sync_layer(args.nets, args.n, args.j);
    free(ptr);
    return 0;
}

pthread_t sync_layer_in_thread(network **nets, int n, int j)
{
    pthread_t thread;
    sync_args *ptr = (sync_args *)calloc(1, sizeof(sync_args));
    ptr->nets = nets;
    ptr->n = n;
    ptr->j = j;
    if(pthread_create(&thread, 0, sync_layer_thread, ptr)) error("Thread creation failed");
    return thread;
}

void sync_nets(network **nets, int n, int interval)
{
    int j;
    int layers = nets[0]->n;
    pthread_t *threads = (pthread_t *) calloc(layers, sizeof(pthread_t));

    *(nets[0]->seen) += interval * (n-1) * nets[0]->batch * nets[0]->subdivisions;
    for (j = 0; j < n; ++j){
        *(nets[j]->seen) = *(nets[0]->seen);
    }
    for (j = 0; j < layers; ++j) {
        threads[j] = sync_layer_in_thread(nets, n, j);
    }
    for (j = 0; j < layers; ++j) {
        pthread_join(threads[j], 0);
    }
    free(threads);
}

float train_networks(network **nets, int n, data d, int interval)
{
    int i;
    int batch = nets[0]->batch;
    int subdivisions = nets[0]->subdivisions;
    assert(batch * subdivisions * n == d.X.rows);
    pthread_t *threads = (pthread_t *) calloc(n, sizeof(pthread_t));
    float *errors = (float *) calloc(n, sizeof(float));

    float sum = 0;
    for(i = 0; i < n; ++i){
        data p = get_data_part(d, i, n);
        threads[i] = train_network_in_thread(nets[i], p, errors + i);
    }
    for(i = 0; i < n; ++i){
        pthread_join(threads[i], 0);
        //printf("%f\n", errors[i]);
        sum += errors[i];
    }
    //cudaDeviceSynchronize();
    if (get_current_batch(nets[0]) % interval == 0) {
        printf("Syncing... ");
        fflush(stdout);
        sync_nets(nets, n, interval);
        printf("Done!\n");
    }
    //cudaDeviceSynchronize();
    free(threads);
    free(errors);
    return (float)sum/(n);
}

void pull_network_output(network *net)
{
    layer l = get_network_output_layer(net);
    cuda_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
}

#endif
