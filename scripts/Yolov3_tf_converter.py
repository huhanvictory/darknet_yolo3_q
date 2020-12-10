import bisect
from collections import defaultdict
from enum import Enum

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.io.gfile import GFile


print(tf.__version__)
# Convolution = namedtuple('Convolution', ['batch_normalize', 'filters', 'size', 'stride', 'pad', 'activation', 'ipos', 'wpos', 'opos',])
# Shortcut = namedtuple('Shortcut', ['from','activation', 'ipos1', 'ipos2', 'opos', 'type', 'name'])
# Upsample = namedtuple('Upsample', ['ipos', 'opos', 'type', 'name'])
# Route = namedtuple('Route', [''])

class DarknetLayer(Enum):
    Convolution = 1
    Shortcut = 2
    Route = 3
    Upsample = 4
    Yolo = 5
    Unknown = -1


def get_conf_header():
    return """[net]
# Testing
# batch=1
# subdivisions=1
# Training
batch=1
subdivisions=1
width=416
height=416
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

learning_rate=0.001
burn_in=1000
max_batches = 500200
policy=steps
steps=400000,450000
scales=.1,.1\n\n"""

def get_yolo1():
    return """[yolo]
mask = 6,7,8
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=20
num=9
jitter=.3
ignore_thresh = .5
truth_thresh = 1
random=1
ipos_bw=8
ipos=2
opos_bw=8
opos=0

[route]
layers = -4
ipos1_bw=8
ipos1=4
opos_bw=8
opos=4

"""

def get_yolo2():
    return """[yolo]
mask = 3,4,5
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=20
num=9
jitter=.3
ignore_thresh = .5
truth_thresh = 1
random=1
ipos_bw=8
ipos=2
opos_bw=8
opos=0

[route]
layers = -4
ipos_bw=8
ipos=4
opos_bw=8
opos=4

"""

def get_yolo3():
    return """[yolo]
mask = 0,1,2
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=20
num=9
jitter=.3
ignore_thresh = .5
truth_thresh = 1
random=1
ipos_bw=8
ipos=2
opos_bw=8
opos=0   
"""


def get_node_index(name):
    return int(name.split('/')[0].split('_')[-1])

name_to_op = {}
conv_index_to_op = defaultdict(list)
add_max_input_to_op = {}
resize_input_to_op = {}
concat_input_to_op = {}

DARKNET_CONF_PATH = './model/gen/Yolov3_q.cfg'  #  path to store the 
DARKNET_WEIGHT_PATH = './model/gen/Yolov3_q.weights'
GRAPH_PB_PATH = './model/deploy_model.pb'

max_layer_index = -1

print("loading graph! ....")
with GFile(GRAPH_PB_PATH, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

for node in graph_def.node:
    node_name = node.name
    name_to_op[node_name] = node

    try:
        node_index = get_node_index(node_name)
    except:
        print('could not get the node index for: ', node_name)
        exit(-1)

    print('Working on node {} with index {} with op {}'.format(node_name, node_index, node.op))
    # check if this node belongs to a convolution layer
    if node.op == 'LeakyReLU' or node.op == 'Conv2D':
        conv_index_to_op[node_index].append(node)
    elif node.op == 'Add':
        inp1 = node.input[0]
        inp1 = get_node_index(inp1)

        inp2 = node.input[1]
        inp2 = get_node_index(inp2)

        add_max_input_to_op[max(inp1, inp2)] = (node, min(inp1, inp2))
    elif node.op == 'DeephiResize':
        inp = node.input[0]
        inp = get_node_index(inp)
        inp = inp + (1 if node.name == 'up_sampling2d_1/ResizeNearestNeighbor' else 2)
        resize_input_to_op[inp] = node
    elif node.op == 'ConcatV2':
        inp = 60 if node.name == 'concatenate_1/concat' else 68
        concat_input_to_op[inp] = node    
    
    if node.op == 'LeakyReLU':  #  checking if leeky relu ipos and opos are the same
        leaky_ipos = node.attr['ipos'].list.i[1]
        leaky_opos = node.attr['opos'].list.i[1]
        assert leaky_ipos == leaky_opos, 'wrong ipos, opos in {}'.format(node_name)
    elif node.op == 'Pad':
        pad_ipos = node.attr['ipos'].list.i[1]
        pad_opos = node.attr['opos'].list.i[1]
        assert pad_ipos == pad_opos, 'wrong ipos, opos in {}'.format(node_name)

    max_layer_index = max(node_index, max_layer_index)

output_layers = [58, 66, 74]
with open(DARKNET_CONF_PATH, 'w') as cfg_f, open(DARKNET_WEIGHT_PATH, 'wb') as wgt_f:
    # writing the header of the DarkNet config file
    header_str = get_conf_header()  
    cfg_f.write(header_str)
    revision = np.array([0, 2, 0, 0, 0], dtype=np.int32)
    revision.tofile(wgt_f)

    for i in range(max_layer_index):
        print('writing weight data of layer: ', i)
        if i + 1 not in conv_index_to_op:  # layers start form 1
            print('layer number %d is missing', i + 1)
            exit(-1)

        # we always push conv first to the list
        conv_layer = conv_index_to_op[i + 1][0]
        cfg_f.write('[convolutional]\n')
        filters_count = conv_layer.attr['bias'].tensor.tensor_shape.dim[0].size
        cfg_f.write('filters={}\n'.format(filters_count))
        filter_size = conv_layer.attr['weights'].tensor.tensor_shape.dim[0].size
        cfg_f.write('size={}\n'.format(filter_size))
        strides = conv_layer.attr['strides'].list.i
        cfg_f.write('stride={}\n'.format(strides[1]))
        cfg_f.write('pad=1\n')  # all convolutions have padding of 1 in YOLOv3
        if i in output_layers:
            cfg_f.write('activation=linear\n')
        else:
            cfg_f.write('activation=leaky\n')

        # writng bias pos 
        bpos_bw, bpos = conv_layer.attr['bpos'].list.i            
        cfg_f.write('bpos_bw={}\n'.format(bpos_bw))
        cfg_f.write('bpos={}\n'.format(bpos))

        # writng weights pos 
        wpos_bw, wpos = conv_layer.attr['wpos'].list.i            
        cfg_f.write('wpos_bw={}\n'.format(wpos_bw))
        cfg_f.write('wpos={}\n'.format(wpos))

        # writng input pos 
        ipos_bw, ipos = conv_layer.attr['ipos'].list.i            
        cfg_f.write('ipos_bw={}\n'.format(ipos_bw))
        cfg_f.write('ipos={}\n'.format(ipos))

        assert (wpos + ipos) >= bpos, 'miss understanding in Conv cond.1'
        

        # writng output pos 
        opos_bw, opos = conv_layer.attr['opos'].list.i            
        cfg_f.write('opos_bw={}\n'.format(opos_bw))
        cfg_f.write('opos={}\n'.format(opos))

        assert (wpos + ipos) >=  opos, 'miss understanding in Conv cond.2'
        cfg_f.write('\n')

        # Handling Add operation
        if(i + 1) in add_max_input_to_op:  # we should insert an Add op here
            add_op, far_input = add_max_input_to_op[(i + 1)]
            cfg_f.write('[shortcut]\n')
            # cfg_f.write('from={}\n'.format(far_input - (i + 1) - 1))
            cfg_f.write('from=-3\n')
            cfg_f.write('activation=linear\n')

            iposes = add_op.attr['ipos'].list.i 
            cfg_f.write('far_ipos_bw={}\n'.format(iposes[0]))
            cfg_f.write('far_ipos={}\n'.format(iposes[1]))
            cfg_f.write('near_ipos_bw={}\n'.format(iposes[2]))
            cfg_f.write('near_ipos={}\n'.format(iposes[3]))
            opos = add_op.attr['opos'].list.i
            cfg_f.write('opos_bw={}\n'.format(opos[0]))
            cfg_f.write('opos={}\n'.format(opos[1]))
            cfg_f.write('\n')

        if (i + 1) in resize_input_to_op:  # we should add resize node here
            upsample_op = resize_input_to_op[i + 1]
            cfg_f.write('[upsample]\n')
            cfg_f.write('stride=2\n')
            ipos_upsample = upsample_op.attr['ipos'].list.i
            cfg_f.write('ipos_bw={}\n'.format(ipos_upsample[0]))
            cfg_f.write('ipos={}\n'.format(ipos_upsample[1]))
            opos = upsample_op.attr['opos'].list.i
            cfg_f.write('opos_bw={}\n'.format(opos[0]))
            cfg_f.write('opos={}\n'.format(opos[1]))
            cfg_f.write('\n')

        if i + 1 == 59:
            cfg_f.write(get_yolo1())

        if i + 1 ==  60:
            cfg_f.write('[route]\n')
            cfg_f.write('layers= -1, 61\n')
            cfg_f.write('ipos1_bw=8\n')
            cfg_f.write('ipos1=4\n')
            cfg_f.write('ipos2_bw=8\n')
            cfg_f.write('ipos2=4\n')
            cfg_f.write('opos_bw=8\n')
            cfg_f.write('opos=4\n')
            cfg_f.write('\n')

        if i + 1 ==  67:
            cfg_f.write(get_yolo2())

        if i + 1 ==  68:
            cfg_f.write('[route]\n')
            cfg_f.write('layers= -1, 36\n')
            cfg_f.write('ipos1_bw=8\n')
            cfg_f.write('ipos1=4\n')
            cfg_f.write('ipos2_bw=8\n')
            cfg_f.write('ipos2=4\n')
            cfg_f.write('opos_bw=8\n')
            cfg_f.write('opos=4\n')
            cfg_f.write('\n')
        if i + 1 ==  75:
            cfg_f.write(get_yolo3())  

        print('Writing weight and bias data in weight file')
        bias_bytes = conv_layer.attr['bias'].tensor.tensor_content
        bias_values = np.frombuffer(bias_bytes, np.float32, filters_count)
        bias_values = bias_values * pow(2, wpos + ipos)
        bias_values = bias_values.astype(np.int32)
        bias_values.tofile(wgt_f)  # writing bias values

        conv_tensor_shape = [conv_layer.attr['weights'].tensor.tensor_shape.dim[i].size for i in range(4)]  # N * H * W * C
        weight_array = np.frombuffer(conv_layer.attr['weights'].tensor.tensor_content, np.float32, np.prod(conv_tensor_shape))
        weight_array = np.reshape(weight_array, conv_tensor_shape)
        weight_array = np.transpose(weight_array, [3, 2, 0, 1])  # this is the way DarkNet keeps the data
        weight_array = weight_array * pow(2, wpos)
        weight_array = weight_array.astype(np.int8)
        #weight_array = np.transpose(weight_array, [3, 0, 1, 2])  # this is the way DarkNet keeps the data
        weight_array.tofile(wgt_f)


print('Done!!!')

        

