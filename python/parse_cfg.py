#!/bin/python
# -*- coding:utf-8 -*-
from __future__ import print_function
import os
import argparse
import shutil
import sys
import logging
import traceback
import json
import re

def dump_to_json(json_file, content, mode="default"):
    '''
    dump json file
    '''
    with open(json_file, 'w') as fileh:
        if mode == "default":
            json.dump(content, fileh, sort_keys=True, indent=4)
        elif mode == "nosort":
            json.dump(content, fileh, sort_keys=False, indent=4)

def update_index_to_string(index):
    str_index = "layer000"
    if index < 10:
        str_index = "00" + str(index)
    elif index < 100:
        str_index = "0" + str(index)
    else:
        str_index = str(index)
    return "layer" + str_index
                                
def get_next_layer(input_layer):
    index = input_layer[-3:]
    next_index = int(index) + 1
    next_layer = update_index_to_string(next_index)
    #print("index input = " + input_layer + ", output = " + str(next_layer))
    return next_layer

class CfgParser:
    """ Helper class for analyzing cfg file """
    def __init__(self, cfg_file):
        self.cfg_file = cfg_file
        with open(self.cfg_file, "r") as f:
            self.lines = f.readlines()

    def get_dict_from_cfg(self, mode):
        dict_network = {}
        index = -1
        dict_name = ""
        for one_line in self.lines:
            regexp = (r"\[(?P<name>\S+)\]")
            match = re.search(regexp, one_line)
            if match:
                (name, ) = match.groups()
                if name == "net":
                    dict_name = "net"
                else:
                    dict_name = update_index_to_string(index)
                dict_network[dict_name] = {}
                dict_network[dict_name]["layer_name"] = name
                index = index + 1
            else:
                regexp2 = (r"(?P<key>\S+)\s*=\s*(?P<value>.*)$")
                match2 = re.search(regexp2, one_line)
                if match2:
                    (key, value) = match2.groups()
                    dict_network[dict_name][key] = value
        return dict_network

class NetworkParser:
    """ Helper class for analyzing cfg file """
    def __init__(self, dict_network):
        self.dict_network = dict_network
        self.map_input_output = {}
        self.map_output_input = {}
        self.map_output_input = {}
        self.index_conv = []
        self.data_offset_in = [] # input and output data address offset
        self.data_offset_out = [] # input and output data address offset
        self.filter_offset = [] # input and output data address offset
        self.input_size = []
        self.output_size = []
        self.filter_size = []
        self.bias_size = []
        self.input_layer_num = 0
        self.output_layer_num = 0

    def generate_data_offset_in(self):
        return self.data_offset_in
    
    def generate_data_offset_out(self):
        return self.data_offset_out
    
    def generate_filter_offset(self):
        return self.filter_offset
    
    def generate_in_out_map(self):
        return self.map_input_output
    
    def generate_out_in_map(self):
        return self.map_output_input
    
    def generate_index_conv(self):
        return self.index_conv
    
    def generate_input_size(self):
        return self.input_size
    
    def generate_output_size(self):
        return self.output_size

    def generate_filter_size(self):
        return self.filter_size

    def generate_bias_size(self):
        return self.bias_size

    def get_input_layer_num(self):
        return self.input_layer_num

    def get_output_layer_num(self):
        return self.output_layer_num

    def generate_config_list(self, image_size, N16xh, PARALLEL_FILTER, SPLITING_FACTOR, ORG_DATA_WIDTH, WIDE_BUS_WIDTH, FACTORS):
        config_dict = {}
        dict_config = {}
        index = 0
        str_index = "000"
        layer_index = "layer" + str_index
        feature_size = image_size
        channel_size = 3
        out_index = -1
        str_out_index = "000"
        out_layer_index = "out" + str(out_index)
        sub_dict = {}
        route_channels = 0
        route_offset = 0
        route_number = 0
        # set the layer and offset and length, which may concat with other layers
        route_reset_layer = 0
        route_reset_offset = 0
        route_reset_size = 0
        data_offset_address = 0
        filter_offset_address = 0
        bias_offset_address = int(75*1024*1024*ORG_DATA_WIDTH/WIDE_BUS_WIDTH)
        #bias_offset_address = 256*1024*1024
        data_offset_index = 0
        global_in_offset = 0
        global_out_offset = 0
        while layer_index in self.dict_network:
            # sub_list
            # 0-3   | enable    | in_w      | in_h      | in_c
            # 4-7   | batch     | out_w     | out_h     | out_c
            # 8-11  | new_w     | stride    | pad       | n
            # 12-15 | bn        | group     | active    | to_ddr
            # 16-19 | w_h_in    | w_h_c_in  | w_h_out   | w_h_c_out
            # 20-23 | split_h   |cond       | bst_w     | new_h_2
            # 24-27 |io_bst_img |  o_lop_bd | 13*o_w    | o_h/13
            # 28-31 |offset_i   |offset_o   | offset_w  | offset_bias
            sub_list = [0] * 32
            layer_name = self.dict_network[layer_index]["layer_name"]
            if layer_name == "convolutional":
                sub_dict = {}
                out_index = out_index + 1
                out_layer_index = update_index_to_string(out_index)
                self.map_output_input[out_layer_index] = ""
                self.index_conv.append(index)

                size = self.dict_network[layer_index]["size"]
                stride = self.dict_network[layer_index]["stride"]
                filters = self.dict_network[layer_index]["filters"]
                pad = self.dict_network[layer_index]["pad"]
                batch_normalize = "0"
                if "batch_normalize" in self.dict_network[layer_index]:
                    batch_normalize = self.dict_network[layer_index]["batch_normalize"]
                activation = self.dict_network[layer_index]["activation"]
                bpos = self.dict_network[layer_index]["bpos"]
                wpos = self.dict_network[layer_index]["wpos"]
                ipos = self.dict_network[layer_index]["ipos"]
                opos = self.dict_network[layer_index]["opos"]
                #print("conv:", layer_index,bpos,wpos,ipos,opos, pos)
                #print(layer_index)
                #0 conv_en / conv size
                sub_list[0] = int(size)
                #1 in_w
                if feature_size % FACTORS == 0:
                    sub_list[1] = int(feature_size)
                else:
                    sub_list[1] = int(feature_size + FACTORS - feature_size % FACTORS)
                #2 in_h
                sub_list[2] = int(feature_size)
                if route_channels > 0:
                    channel_size = route_channels
                    route_channels = 0
                #3 in_c : make sure channel_size is multiples of PARALLEL_FILTER
                sub_list[3] = int((int(channel_size)+PARALLEL_FILTER-1)/PARALLEL_FILTER)*PARALLEL_FILTER
                #4 batch
                sub_list[4] = 1
                if stride == "1":
                    feature_size = feature_size / 1
                elif stride == "2":
                    feature_size = feature_size / 2
                #5 out_w
                sub_list[5] = int(sub_list[1]/int(stride))
                #6 out_h
                sub_list[6] = int(feature_size)
                #7 out_c : make sure filters is multiples of PARALLEL_FILTER
                sub_list[7] = int((int(filters)+PARALLEL_FILTER-1)/PARALLEL_FILTER)*PARALLEL_FILTER
                channel_size = filters
                #8 new_w : new w of image is multiples of FACTORS
                if sub_list[6] % FACTORS == 0:
                    sub_list[8] = int(feature_size)
                else:
                    sub_list[8] = int(feature_size + FACTORS - (feature_size % FACTORS))
                #9 stride
                sub_list[9] = int(stride)
                if size == "1":
                    pad = 0
                #10 pad
                sub_list[10] = int(pad)
                #11 pos : for bias quantization 
                sub_list[11] = int(wpos) + int(ipos) - int(opos)

                #12 batch_normalize or loop_3 : stride 1 = w+1; stride 2 = w/2
                #sub_list[12] = int(batch_normalize)
                sub_list[12] = int((sub_list[1] + sub_list[10] - sub_list[9] + 1)/sub_list[9])
                #13
                sub_list[13] = 0
                relu = 0
                if activation == "leaky":
                    relu = 1
                #14 activation
                sub_list[14] = int(relu)
                #15
                sub_list[15] = 0
                #16 in_w * in_h
                sub_list[16] = sub_list[1] * sub_list[2]
                #17 in_w * in_h * in_c
                sub_list[17] = sub_list[1] * sub_list[2] * sub_list[3]
                #18 out_w * out_h
                sub_list[18] = sub_list[5] * sub_list[6]
                #19 out_w * out_h * out_c
                sub_list[19] = sub_list[5] * sub_list[6] * sub_list[7]
                #20 split_h: #should be the divisor of current h
                if sub_list[2] == 416:#13x32
                    sub_list[20] = SPLITING_FACTOR
                elif sub_list[2] == 208:#13x16
                    if sub_list[9] == 1:#stride=1
                        sub_list[20] = SPLITING_FACTOR * 2
                    else:#stride=2 
                        sub_list[20] = SPLITING_FACTOR * 2 + 1 #fully use burst in ping-pang bram, must be odd if not eqal h
                elif sub_list[2] == 104:
                    if sub_list[9] == 1:#stride=1
                        sub_list[20] = SPLITING_FACTOR * 4
                    else:
                        sub_list[20] = SPLITING_FACTOR * 4 + 5 #fully use burst in ping-pang bram, must be odd if not eqal h
                elif sub_list[2] == 52:
                    sub_list[20] = SPLITING_FACTOR * 4
                elif sub_list[2] == 26:
                    sub_list[20] = SPLITING_FACTOR * 2
                else:
                    sub_list[20] = 13
                #21 conv 3x3 cond : block number after image h is split
                sub_list[21] = int(sub_list[2]/(sub_list[20] + sub_list[9] - 1))
                #22 burst_length_filter
                sub_list[22] = int(sub_list[3] * sub_list[0] * sub_list[0] * PARALLEL_FILTER * ORG_DATA_WIDTH / WIDE_BUS_WIDTH)
                #23 new_h_2 : new h of last block after image h is split 
                sub_list[23] = int((sub_list[2] + 2 * sub_list[10] - sub_list[0])%(sub_list[20] + sub_list[9] - 1) + 1)
                #24 burst_length
                #N16xh:
                #52: 52/13=4, 13 image->4*16 = burst 64 planes, 26-> 2*16 = 32
                #104: 104/13=8, 13 image ->8*16 = burst 128 planes
                #208: 208/13=16, 13 image ->16*16 = burst 256 planes
                if(sub_list[2] == 13 or sub_list[2] == 26 or sub_list[2] == 52):
                    sub_list[24] = int(sub_list[1] * N16xh * PARALLEL_FILTER * ORG_DATA_WIDTH / WIDE_BUS_WIDTH)
                #elif(sub_list[2] == 26):
                #    sub_list[24] = int(sub_list[1] * sub_list[20] * PARALLEL_FILTER * ORG_DATA_WIDTH / WIDE_BUS_WIDTH)
                #elif(sub_list[2] == 52):
                #    sub_list[24] = int(sub_list[1] * sub_list[20] * PARALLEL_FILTER * ORG_DATA_WIDTH / WIDE_BUS_WIDTH)
                else:
                    sub_list[24] = int(sub_list[1] * (sub_list[20] + 2 * sub_list[10]) * PARALLEL_FILTER * ORG_DATA_WIDTH / WIDE_BUS_WIDTH)
                #25 w_col : w after conv 
                sub_list[25] = int((sub_list[1] + 2 * sub_list[10] - sub_list[0]) / sub_list[9] + 1)
                #26 in_w * split_h
                sub_list[26] = sub_list[1] * sub_list[20]
                #27 in_h / split_h
                sub_list[27] = int((sub_list[2] + sub_list[20] - 1 + sub_list[9] - 1) / (sub_list[20] + sub_list[9] - 1))
                
                # generate data offset
                sub_list[28] = data_offset_address
                if route_number == 1 and route_offset > 0:
                    sub_list[28] = route_offset
                    #print("conv oute_offset = " + str(route_offset))
                    route_offset = 0
                global_in_offset = sub_list[28]
                self.data_offset_in.append(data_offset_address)
                data_offset_address += int(sub_list[17] * ORG_DATA_WIDTH / WIDE_BUS_WIDTH)
                sub_list[29] = data_offset_address 
                global_out_offset = sub_list[29]
                # reset previous layer data out offset address
                # reset previous layer's next layer data in offset address
                if route_number == 2 and route_offset > 0:
                    route_offset = 0
                self.data_offset_out.append(data_offset_address)
                sub_list[30] = filter_offset_address
                self.filter_offset.append(filter_offset_address)
                filter_offset_address += int(sub_list[3] * sub_list[0] * sub_list[0] * sub_list[7] * ORG_DATA_WIDTH / WIDE_BUS_WIDTH)
                sub_list[31] = bias_offset_address
                bias_offset_address += int(1024*32/WIDE_BUS_WIDTH)

                self.input_size.append(int(sub_list[1] * sub_list[2] * sub_list[3] * ORG_DATA_WIDTH / WIDE_BUS_WIDTH))
                self.output_size.append(int(sub_list[8] * sub_list[6] * sub_list[7] * ORG_DATA_WIDTH / WIDE_BUS_WIDTH))
                self.filter_size.append(int(sub_list[3] * sub_list[7] * sub_list[0] * sub_list[0] * ORG_DATA_WIDTH / WIDE_BUS_WIDTH))
                self.bias_size.append(int(1024*32/WIDE_BUS_WIDTH))
                    
                # collect all data and fill to layer
                sub_dict[layer_name] = sub_list

            elif layer_name == "shortcut":
                from_layer = self.dict_network[layer_index]["from"]
                from_cnt = int(from_layer)
                activation = self.dict_network[layer_index]["activation"]
                far_ipos = self.dict_network[layer_index]["far_ipos"]
                near_ipos = self.dict_network[layer_index]["near_ipos"]
                if int(far_ipos) > int(near_ipos):
                    max_ipos = int(far_ipos)
                else:
                    max_ipos = int(near_ipos)
                opos = self.dict_network[layer_index]["opos"]
                pos = (max_ipos - int(far_ipos))*(2^16) + (max_ipos - int(near_ipos))*(2^8) + (max_ipos - int(opos))
                #print("shortcut:",layer_index, far_ipos, near_ipos, max_ipos, opos, pos)
                #0 conv_en / conv size
                sub_list[0] = int(1)
                #1 in_w
                if feature_size % FACTORS == 0:
                    sub_list[1] = int(feature_size)
                else:
                    sub_list[1] = int(feature_size + FACTORS - (feature_size % FACTORS))
                #2 in_h
                sub_list[2] = int(feature_size)
                #3 in_c
                sub_list[3] = int((int(channel_size)+PARALLEL_FILTER-1)/PARALLEL_FILTER)*PARALLEL_FILTER
                #4 batch void
                sub_list[4] = 0
                #5 out_w
                if feature_size % FACTORS == 0:
                    sub_list[5] = int(feature_size)
                else:
                    sub_list[5] = int(feature_size + FACTORS - (feature_size % FACTORS))
                #6 out_h
                sub_list[6] = int(feature_size)
                #7 out_c
                sub_list[7] = int((int(filters)+PARALLEL_FILTER-1)/PARALLEL_FILTER)*PARALLEL_FILTER
                #8 new_w
                if sub_list[6] % FACTORS == 0:
                    sub_list[8] = int(feature_size)
                else:
                    sub_list[8] = int(feature_size + FACTORS - (feature_size % FACTORS))
                #11 pos
                sub_list[11] = max_ipos - int(near_ipos)
                #12 pos
                sub_list[12] = max_ipos - int(far_ipos)
                #13 pos
                sub_list[13] = max_ipos - int(opos)
                #15
                sub_list[15] = 0
                #16 in_w * in_h
                sub_list[16] = sub_list[1] * sub_list[2]
                sub_list[17] = sub_list[1] * sub_list[2] * sub_list[3]
                sub_list[18] = sub_list[5] * sub_list[6]
                sub_list[19] = sub_list[5] * sub_list[6] * sub_list[7]
                #24 burst_length
                if(sub_list[1] == 28):
                    sub_list[24] = int(13 * sub_list[1] *  8)
                elif(sub_list[1] ==16 or sub_list[1] == 14):
                    sub_list[24] = int(13 * sub_list[1] * 16)
                else:
                    sub_list[24] = int(13 * 208)
                #25 loop_bound
                sub_list[25] = int(sub_list[5] * sub_list[6] * sub_list[7] / PARALLEL_FILTER / sub_list[24])
                sub_list[26] = sub_list[1] * sub_list[20]
                sub_list[28] = global_in_offset
                sub_list[29] = global_out_offset
                # set shortcut adder address
                if from_cnt < 0:
                    from_layer = update_index_to_string(index + from_cnt)
                else:
                    from_layer = update_index_to_string(from_cnt)
                layer_str = self.map_input_output[from_layer]
                length = len(config_dict[layer_str])
                i = 0
                for e in config_dict[layer_str]:
                    if i == length - 1:
                        sub_list[28] = int(config_dict[layer_str][e][29])
                        #print("channels = " + str(route_channels))
                    i += 1
                sub_dict[layer_name] = sub_list

            elif layer_name == "route":
                layers = self.dict_network[layer_index]["layers"]
                #current only support route number=2
                route_layer = layers.split(",")
                route_number = len(route_layer)
                #print("route number = " + str(route_number))
                route_layer_index = 0
                for one_layer in route_layer:
                    layer_int = int(one_layer)
                    if layer_int <= 0:
                        layer_int = index + layer_int
                    layer_int_index = update_index_to_string(layer_int)
                    route_out_index = self.map_input_output[layer_int_index]
                    if route_layer_index == 1:
                        route_reset_layer = route_out_index
                    #print("route_out_index = " + route_out_index)
                    length = len(config_dict[route_out_index])
                    i = 0
                    for e in config_dict[route_out_index]:
                        if route_number == 1:
                            route_offset = int(config_dict[route_out_index][e][29])
                            #if config_dict[route_out_index][e][6] == 13 or config_dict[route_out_index][e][6] == 26:
                            #    config_dict[route_out_index][e][15] = 2
                            #else:
                            #    config_dict[route_out_index][e][15] = 1
                        if i == length - 1:
                            tmp = int(config_dict[route_out_index][e][7])
                            route_channels += tmp
                            #print("channels = " + str(route_channels))
                            if route_number == 2 and route_layer_index == 0:
                                # get offset and size for target route layer(the second number of from layers)
                                route_reset_offset = int(config_dict[route_out_index][e][29])
                                route_reset_size = int(config_dict[route_out_index][e][8] * config_dict[route_out_index][e][6] * config_dict[route_out_index][e][7] * ORG_DATA_WIDTH / WIDE_BUS_WIDTH)
                                #print("route_reset_offset = " + str(route_reset_offset))
                                #print("route_reset_size = " + str(route_reset_size))
                            if route_number == 2 and route_layer_index == 1:
                                # reset target route output address(the second number of from layers)
                                route_reset_address = route_reset_offset + route_reset_size 
                                #print("route_reset_address = " + str(route_reset_address))
                                config_dict[route_out_index][e][29] = route_reset_address
                                # reset target route input address(next layer of the second number of from layers)
                                next_index = get_next_layer(route_out_index)
                                config_dict[next_index]["convolutional"][28] = route_reset_address
                        i += 1
                    route_layer_index += 1
                    #print("layers = " + str(layer_int))

            elif layer_name == "upsample":
                stride = self.dict_network[layer_index]["stride"]
                #0 conv_en / conv size
                sub_list[0] = int(1)
                #1 in_w
                if feature_size % FACTORS == 0:
                    sub_list[1] = int(feature_size)
                else:
                    sub_list[1] = int(feature_size + FACTORS - (feature_size % FACTORS))
                #2 in_h
                sub_list[2] = int(feature_size) 
                #3 in_c
                sub_list[3] = int((int(channel_size)+PARALLEL_FILTER-1)/PARALLEL_FILTER)*PARALLEL_FILTER
                feature_size = feature_size * 2
                #4
                #5 out_w
                sub_list[5] = sub_list[1]*2
                #sub_list[5] = int(feature_size + FACTORS - (feature_size % FACTORS))
                #6 out_h
                sub_list[6] = int(feature_size)
                #7 out_c
                sub_list[7] = int((int(filters)+PARALLEL_FILTER-1)/PARALLEL_FILTER)*PARALLEL_FILTER
                #8 new_w
                if sub_list[6] % FACTORS == 0:
                    sub_list[8] = int(feature_size)
                else:
                    sub_list[8] = int(feature_size + FACTORS - (feature_size % FACTORS))
                #15
                sub_list[15] = 0
                #16 in_w * in_h
                sub_list[16] = sub_list[1] * sub_list[2]
                #17 in_w * in_h * in_c
                sub_list[17] = sub_list[1] * sub_list[2] * sub_list[3]
                #18 out_w * out_h
                sub_list[18] = sub_list[5] * sub_list[6]
                #19 out_w * out_h * out_c
                sub_list[19] = sub_list[5] * sub_list[6] * sub_list[7]
                #20 read_fifo_line
                if(index == 58 or index == 66 or index == 74):
                    sub_list[20] = int(13 * 8)
                elif(sub_list[5] == 56):
                    sub_list[20] = int(13 * 8)
                elif(sub_list[5] == 28 or sub_list[5] == 26 or sub_list[5] == 32 or sub_list[5] == 16 or sub_list[5] == 14):
                    sub_list[20] = int(13 * 16)
                else:
                    sub_list[20] = int(13 * 416 / sub_list[5])
                #24 burst_length
                if(index == 58 or index == 66 or index == 74):
                    sub_list[24] = int(13 * sub_list[8] * 5 * PARALLEL_FILTER * ORG_DATA_WIDTH / WIDE_BUS_WIDTH)
                elif(sub_list[5] == 56):
                    sub_list[24] = int(13 * sub_list[8] * 8 * PARALLEL_FILTER * ORG_DATA_WIDTH / WIDE_BUS_WIDTH)
                elif(sub_list[8] == 28 or sub_list[8] == 16 or sub_list[5] == 14):
                    sub_list[24] = int(13 * sub_list[8] * 16 * PARALLEL_FILTER * ORG_DATA_WIDTH / WIDE_BUS_WIDTH)
                else:
                    sub_list[24] = int(13 * 416  * PARALLEL_FILTER * ORG_DATA_WIDTH / WIDE_BUS_WIDTH)
                #25 loop_bound
                sub_list[25] = int(sub_list[5] * sub_list[6] * sub_list[7] * ORG_DATA_WIDTH / WIDE_BUS_WIDTH / sub_list[24])
                #26 out_h_4
                sub_list[26] = int(sub_list[6] % FACTORS)
                #27 
                sub_list[27] = 0 
                sub_list[28] = global_in_offset
                sub_list[29] = global_out_offset
                sub_dict[layer_name] = sub_list
                self.input_size.append(sub_list[1] * sub_list[2] * sub_list[3] * ORG_DATA_WIDTH / WIDE_BUS_WIDTH)
                self.output_size.append(sub_list[8] * sub_list[6] * sub_list[7] * ORG_DATA_WIDTH / WIDE_BUS_WIDTH)

            elif layer_name == "yolo":
                #previous_index = update_index_to_string(index - 1)
                #previous_out_index = self.map_input_output[previous_index]
                #length = len(config_dict[previous_out_index])
                #i = 0
                #for e in config_dict[previous_out_index]:
                #    if i == length - 1:
                #        print(i)
                #        config_dict[previous_out_index][e][15] = 1
                #    else:
                #        config_dict[previous_out_index][e][15] = 0
                #    i += 1
                pass
            config_dict[out_layer_index] = sub_dict

            self.map_input_output[layer_index] = out_layer_index
            self.map_output_input[out_layer_index] += " " + layer_index
            index = index + 1
            layer_index = update_index_to_string(index)
        #print(self.map_input_output)
        #print(data_offset_in)
        #print(data_offset_out)
        self.input_layer_num = index
        self.output_layer_num = out_index + 1
        #print(config_dict)
        return config_dict

class CodeGen:
    """ Helper class for analyzing cfg file """
    def __init__(self, map_input_output, map_output_input):
        self.map_input_output = map_input_output
        self.map_output_input = map_output_input
        pass

    def generate_offset_header(self, var_name, data_offset_in):
        header = ""
        header += "const int " + var_name + "[" + str(len(data_offset_in)) + "] = {\n"
        for offset in data_offset_in:
            header += "\t" + str(offset)
            if data_offset_in.index(offset) != len(data_offset_in) - 1:
                header += ", //layer " + str(data_offset_in.index(offset)) + "\n"
            else:
                header += " //layer " + str(data_offset_in.index(offset)) + "\n"
        header += "};\n"
        return header

    def generate_config_list_header(self, dict_config):
        header = ""
        dict_len = len(dict_config)
        header_config_list = ""
        header_config_list += "const int config_list_all[" + str(dict_len) + "][3][32] = {\n"
        config_dict = {}
        index = 0
        str_index = "000"
        layer_index = "layer" + str_index
        while layer_index in dict_config:
            header_config_list += "\t{ //" + layer_index + " =>" + self.map_output_input[layer_index] + "\n"
            config_layer = dict_config[layer_index]
            layer_name = ["convolutional", "shortcut", "upsample"]
            conv_list = [0] * 32
            count_layer = 0
            to_ddr = 0
            for one_layer in layer_name:
                if one_layer in dict_config[layer_index]:
                    sub_config = dict_config[layer_index][one_layer]
                    conv_list = sub_config
                    header_config_list += "\t\t{"
                    count = 0
                    for e in sub_config:
                        header_config_list += str(e)
                        if count < 31:
                            header_config_list += ","
                        count += 1
                    header_config_list += "},\n"
                else:
                    default_list = [0] * 32
                    default_list[1] = conv_list[5] 
                    default_list[2] = conv_list[6] 
                    default_list[3] = conv_list[7] 
                    default_list[5] = conv_list[5] 
                    default_list[6] = conv_list[6] 
                    default_list[7] = conv_list[7] 
                    default_list[8] = conv_list[8] 
                    #15
                    default_list[15] = conv_list[15] 
                    default_list[16] = default_list[1] * default_list[2]
                    default_list[17] = default_list[1] * default_list[2] * default_list[3]
                    default_list[18] = default_list[5] * default_list[6]
                    default_list[19] = default_list[5] * default_list[6] * default_list[7]
                    #20 read_fifo_line
                    if(index == 58 or index == 66 or index == 74):
                        default_list[20] = int(13 * 5)
                    elif(default_list[5] == 56):
                        default_list[20] = int(13 * 8)
                    elif(default_list[5] == 28 or default_list[5] == 26 or default_list[5] == 32 or default_list[5] == 16 or default_list[5] == 14):
                        default_list[20] = int(13 * 16)
                    else:
                        default_list[20] = int(13 * 416 / default_list[5])
                    default_list[21] = conv_list[21] 
                    #24 burst_length
                    if(index == 58 or index == 66 or index == 74):
                        default_list[24] = int(13 * conv_list[8] * 5 * PARALLEL_FILTER * ORG_DATA_WIDTH / WIDE_BUS_WIDTH)
                    elif(conv_list[5] == 56):
                        default_list[24] = int(13 * conv_list[8] * 8 * PARALLEL_FILTER * ORG_DATA_WIDTH / WIDE_BUS_WIDTH)
                    elif(conv_list[8] == 28 or conv_list[8] == 16 or conv_list[5] == 14):
                        default_list[24] = int(13 * conv_list[8] * 16 * PARALLEL_FILTER * ORG_DATA_WIDTH / WIDE_BUS_WIDTH)
                    else:
                        default_list[24] = int(13 * 416  * PARALLEL_FILTER * ORG_DATA_WIDTH / WIDE_BUS_WIDTH)
                    #25 loop_bound
                    default_list[25] = int(conv_list[8] * conv_list[6] * conv_list[7] * ORG_DATA_WIDTH / WIDE_BUS_WIDTH / default_list[24])
                    #26 out_h_4
                    default_list[26] = int(conv_list[6] % FACTORS)
                    default_list[27] = conv_list[27] 
                    default_list[28] = conv_list[28] 
                    default_list[29] = conv_list[29] 
                    default_list[30] = conv_list[30] 
                    default_list[31] = conv_list[31] 
                    header_config_list += "\t\t{"
                    count = 0
                    for e in default_list:
                        header_config_list += str(e)
                        if count < 31:
                            header_config_list += ","
                        count += 1
                    header_config_list += "}"
                    if count_layer < 2:
                        header_config_list += ","
                    header_config_list += "\n"
                count_layer += 1
            header_config_list += "\t}"
            if index < dict_len - 1:
                header_config_list += ","
            index = index + 1
            header_config_list += "\n"
            layer_index = update_index_to_string(index)
        header_config_list += "};\n"
        header += header_config_list
        return header

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="input configuration file")
    parser.add_argument("--N16xh", required=True, help="input burst line:N16 x in_h")
    parser.add_argument("--image_size", required=False, help="input image size")
    args = parser.parse_args()
    N16xh = int(args.N16xh)
    image_size = args.image_size
    image_size = 416
    SPLITING_FACTOR = 13
    PARALLEL_FILTER = 16
    TILING_IMAGE = 16
    ORG_DATA_WIDTH = 8
    WIDE_BUS_WIDTH = 512
    FACTORS = int(WIDE_BUS_WIDTH / ORG_DATA_WIDTH / PARALLEL_FILTER)
    cfg_file = args.cfg
    print("1. Parsing cfg file...")
    cfg_parser = CfgParser(cfg_file)
    dict_cfg = cfg_parser.get_dict_from_cfg("1")
    dump_to_json("config_parser.json", dict_cfg)
    print("   config_parser.json generated.")
    print("2. Parsing network...")
    network_parser = NetworkParser(dict_cfg)
    dict_config = network_parser.generate_config_list(image_size, N16xh, PARALLEL_FILTER, SPLITING_FACTOR, ORG_DATA_WIDTH, WIDE_BUS_WIDTH, FACTORS)
    data_offset_in = network_parser.generate_data_offset_in()
    data_offset_out = network_parser.generate_data_offset_out()
    filter_offset = network_parser.generate_filter_offset()
    map_input_output = network_parser.generate_in_out_map()
    map_output_input = network_parser.generate_out_in_map()
    input_size = network_parser.generate_input_size()
    output_size = network_parser.generate_output_size()
    filter_size = network_parser.generate_filter_size()
    bias_size = network_parser.generate_bias_size()
    index_conv = network_parser.generate_index_conv()
    input_layer_num = network_parser.get_input_layer_num()
    output_layer_num = network_parser.get_output_layer_num()
    dump_to_json("config_dict.json", dict_config)
    print("   config_dict.json generated.")
    print("3. Generating header file...")
    code_gen = CodeGen(map_input_output, map_output_input)
    #generate config list
    header_config = ""
    #add constant definition
    header_config += "#define OVERLAP            2\n"
    header_config += "#define COMPUTE_UNIT       1\n"
    header_config += "#define TILING_IMAGE       " + str(TILING_IMAGE) + "\n"
    header_config += "#define PARALLEL_FILTER    " + str(PARALLEL_FILTER) + "\n"
    header_config += "#define ORG_DATA_WIDTH     " + str(ORG_DATA_WIDTH) + "\n"
    header_config += "#define WIDE_BUS_WIDTH     " + str(WIDE_BUS_WIDTH) + "\n"
    header_config += "#define SPLITING_FACTOR    " + str(SPLITING_FACTOR) + "\n"
    header_config += "#define INPUT_LAYER_NUM    " + str(input_layer_num) + "\n"
    header_config += "#define OUTPUT_LAYER_NUM   " + str(output_layer_num) + "\n"
    header_config += "#define FACTORS            " + str(FACTORS) + "\n"

    #add constant array
    header_config += code_gen.generate_config_list_header(dict_config)
    header_config += code_gen.generate_offset_header("in_offset_map", data_offset_in)
    print("   in_offset_map generated.")
    header_config += code_gen.generate_offset_header("out_offset_map", data_offset_out)
    print("   out_offset_map generated.")
    header_config += code_gen.generate_offset_header("weights_offset_map", filter_offset)
    print("   filter_offset_map generated.")
    #header_config += code_gen.generate_offset_header("input_size", input_size)
    #print("   input_size generated.")
    #header_config += code_gen.generate_offset_header("output_size", output_size)
    #print("   output_size generated.")
    #header_config += code_gen.generate_offset_header("filter_size", filter_size)
    #print("   filter_size generated.")
    #header_config += code_gen.generate_offset_header("bias_size", bias_size)
    #print("   bias_size generated.")
    header_config += code_gen.generate_offset_header("index_conv", index_conv)
    print("   index_conv generated.")
    #generate address list
    if PARALLEL_FILTER == 16:
        config_file_name = 'src/config_16x16_q.h';
    elif PARALLEL_FILTER == 8:
        config_file_name = 'src/config_8x8.h';
    elif PARALLEL_FILTER == 4:
        config_file_name = 'src/config_4x4.h';
    else:
        config_file_name = 'src/config.h';


    with open(config_file_name, 'w') as f:
        f.write(header_config)
        print( config_file_name + " generated.")
    sys.exit(0)
