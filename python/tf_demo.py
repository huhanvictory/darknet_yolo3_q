import argparse
from multiprocessing import Queue, Process
from multiprocessing.managers import BaseManager
from collections import deque
from abc import ABC, abstractmethod
from os import listdir, path
from os.path import isfile, join
import random
import time
import threading

import tensorflow.compat.v1 as tf
from numpy.testing import assert_array_equal
import numpy as np
import cv2

def get_color_table(class_num, seed=2):
    random.seed(seed)
    color_table = {}
    for i in range(class_num):
        color_table[i] = [random.randint(0, 255) for _ in range(3)]
    return color_table

def plot_one_box(img, coord, label=None, color=None, line_thickness=None):
    '''
    coord: [x_min, y_min, x_max, y_max] format coordinates.
    img: img to plot on.
    label: str. The label name.
    color: int. color index.
    line_thickness: int. rectangle line thickness.
    '''
    tl = line_thickness or int(round(0.002 * max(img.shape[0:2])))  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

def letterbox_converter(cv_img):
    H, W, _ = cv_img.shape
    r_h, r_w = 416 / H, 416 / W
    r_f = min (r_h, r_w)

    dest_H, dest_W = int(H * r_f), int(W * r_f)
    res_image = cv2.resize(cv_img, (dest_W, dest_H))

    lett_image = np.full(shape=(416, 416, 3), fill_value=127, dtype=np.uint8)

    lett_image[(416 - dest_H) // 2: (416 - dest_H) // 2 + dest_H, (416 - dest_W) // 2:(416 - dest_W) // 2 + dest_W, :] = res_image

    return lett_image


def py_nms(boxes, scores, max_boxes=50, iou_thresh=0.5):
    """
    Pure Python NMS baseline.
    Arguments: boxes: shape of [-1, 4], the value of '-1' means that dont know the
                      exact number of boxes
               scores: shape of [-1,]
               max_boxes: representing the maximum of boxes to be selected by non_max_suppression
               iou_thresh: representing iou_threshold for deciding to keep boxes
    """
    assert boxes.shape[1] == 4 and len(scores.shape) == 1

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]

    return keep[:max_boxes]


def cpu_nms(boxes, scores, num_classes, max_boxes=50, score_thresh=0.2, iou_thresh=0.5):
    """
    Perform NMS on CPU.
    Arguments:
        boxes: shape [1, 10647, 4]
        scores: shape [1, 10647, num_classes]
    """

    boxes = boxes.reshape(-1, 4)
    scores = scores.reshape(-1, num_classes)
    # Picked bounding boxes
    picked_boxes, picked_score, picked_label = [], [], []

    for i in range(num_classes):
        indices = np.where(scores[:, i] >= score_thresh)
        filter_boxes = boxes[indices]
        filter_scores = scores[:, i][indices]
        if len(filter_boxes) == 0:
            continue
        # do non_max_suppression on the cpu
        indices = py_nms(filter_boxes, filter_scores,
                         max_boxes=max_boxes, iou_thresh=iou_thresh)
        picked_boxes.append(filter_boxes[indices])
        picked_score.append(filter_scores[indices])
        picked_label.append(np.ones(len(indices), dtype='int32')*i)
    if len(picked_boxes) == 0:
        return None, None, None

    boxes = np.concatenate(picked_boxes, axis=0)
    score = np.concatenate(picked_score, axis=0)
    label = np.concatenate(picked_label, axis=0)

    return boxes, score, label


class ImageProvider(ABC):
    @abstractmethod
    def __iter__(self):
        pass


class PathImageReader(ImageProvider):
    def __init__(self, path, count=float('inf')):
        self._path = path
        self._images = [join(path, f) for f in listdir(path) if isfile(
            join(path, f))]
        self._count = count

    def __iter__(self):
        curr_count = 0
        for path in self._images:
            curr_count += 1
            org_img = cv2.imread(path)
            img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
            # later we will make this letterbox resize
            resized_img = letterbox_converter(img)
            # resized_img = cv2.resize(img, (416, 416))
            yield resized_img, org_img
            if curr_count == self._count:
                break


class EventRateCalc:
    def __init__(self, q_size=20):
        self._t_q = deque(maxlen=q_size)

    def add_time(self, t):
        self._t_q.append(t)

    def get_rate(self):
        #print ("in reader: ", id(self))
        if len(self._t_q) > 1:
            return (len(self._t_q) - 1) / ((self._t_q[-1] - self._t_q[0]) / 1000000000)
        else:
            return 0

    def tick(self):
        self.add_time(time.time_ns())       


class YoloV3Infer:
    def __init__(self, args, q_max_len=0):
        self._args = args
        self._qs = [Queue(q_max_len)
                    for _ in range(3)]  # we have a 4 stage pipeline

        BaseManager.register('EventRateCalc', EventRateCalc)
        manager = BaseManager()
        manager.start()
        self._rc_measure = [ manager.EventRateCalc() for _ in range(4)]

        self._shut_down = False
        self._presentation_q = Queue(0)  # we keep this q as an unlimiterd size since it will be meters by the pipeline Q

        if args.mode == 'path':
            self._provider = PathImageReader(args.img_path)
        else:
            print(args.mode, "has not been implemented yet!")
            exit(1)  # we should implement the rest of providers here

    def shut_down(self):
        self._shut_down = True

    def print_rates(self):
        CRED    = '\33[31m'
        CGREEN  = '\33[32m'
        CYELLOW = '\33[33m'
        CBLUE   = '\33[34m'
        CEND      = '\33[0m'

        print(CRED + "Pre-processor: {:.2f}".format(self._rc_measure[0].get_rate() * self._args.batch), " iamges per second!" + CEND)    
        print(CGREEN + "Processor: {:.2f}".format(self._rc_measure[1].get_rate() * self._args.batch), " images per second!" + CEND)    
        print(CYELLOW + "Post-processer: {:.2f}".format(self._rc_measure[2].get_rate() * self._args.batch), " images per second!" + CEND)    
        print(CBLUE + "Presenter: {:.2f}".format(self._rc_measure[3].get_rate() * self._args.batch), " images per second!" + CEND)
        print("\n")    

    def start_threads(self):

        t1 = Process(target=self._pre_processor,
                     args=(self._qs[0], self._args.batch, self._presentation_q, self._rc_measure[0]))
        t2 = Process(target=self._processor, args=(
            self._qs[0], self._qs[1], self._args.batch, self._rc_measure[1]))
        t3 = Process(target=self._post_processor, args=(
            self._qs[1], self._qs[2], self._args.batch, self._rc_measure[2]))
        t4 = Process(target=self._presentation, args=(
            self._qs[2], self._args.batch, self._presentation_q, self._rc_measure[3]))

        t1.start()
        t2.start()
        t3.start()
        t4.start()

        return t4  # we return the last thread to the main thread to join

    def _pre_processor(self, out_q, batch, presentation_q, rc):
        curr_batch = np.zeros((batch, 3, 416, 416), dtype=np.uint8)
        for curr_idx, (frame, org_frame) in enumerate(self._provider):
            
            presentation_q.put(org_frame)

            # make it compatible with DarkNet format
            frame = np.transpose(frame, (2, 0, 1))

            # frame = np.zeros((3, 416, 416))
            # for c in range(3):
            #     for h in range(416):
            #         for w in range(416):
            #             frame[c, h, w] = (c + h + w) % 256

            curr_batch[curr_idx % batch, :, :, :] = frame

            if (curr_idx + 1) % batch == 0:
                # print ("in writer: ", id(rc))
                rc.tick()
                out_q.put(curr_batch / 255.0)
                curr_batch = np.zeros((batch, 3, 416, 416), dtype=np.uint8)

            if self._shut_down:
                break    

        if (curr_idx + 1) % batch != 0:
            out_q.put(curr_batch)
        out_q.put('STOP')

    def _processor(self, in_q, out_q, batch, rc):
        tf.reset_default_graph()
        processor_graph = tf.Graph()
        with processor_graph.as_default() as g1:
            with g1.name_scope("processor_graph") as scope:
                op_module = tf.load_op_library('libkernel.so')
                inp_batch = tf.placeholder(dtype=float, shape=(
                    batch, 3, 416, 416), name="place_holder1")
                out_s, out_m, out_l = op_module.My_Yolo_OP(
                    yolo_input=inp_batch)
                init_op = tf.initialize_all_variables()

        with tf.Session(graph=processor_graph) as sess:
            sess.run(init_op)
            while True:
                curr_batch = in_q.get()
                if curr_batch == 'STOP':
                    out_q.put('STOP')
                    break
                yolo_out_s, yolo_out_m, yolo_out_l = sess.run(
                    [out_s, out_m, out_l], feed_dict={inp_batch: curr_batch})
                rc.tick()    
                out_q.put((yolo_out_s, yolo_out_m, yolo_out_l))

    def _post_processor(self, in_q, out_q, batch, rc):
        def reorg_layer(feature_map, anchors):
            '''
            feature_map: a feature_map from [feature_map_1, feature_map_2, feature_map_3] returned
                from `forward` function
            anchors: shape: [3, 2]
            '''
            # NOTE: size in [h, w] format! don't get messed up!
            grid_size = feature_map.get_shape().as_list()[1:3]
            # the downscale ratio in height and weight
            ratio = tf.cast(tf.constant([416, 416]) / grid_size, tf.float32)
            # rescale the anchors to the feature_map
            # NOTE: the anchor is in [w, h] format!
            rescaled_anchors = [
                (anchor[0] / ratio[1], anchor[1] / ratio[0]) for anchor in anchors]

            feature_map = tf.reshape(
                feature_map, [-1, grid_size[0], grid_size[1], 3, 5 + 20])

            # split the feature_map along the last dimension
            # shape info: take 416x416 input image and the 13*13 feature_map for example:
            # box_centers: [N, 13, 13, 3, 2] last_dimension: [center_x, center_y]
            # box_sizes: [N, 13, 13, 3, 2] last_dimension: [width, height]
            # conf_logits: [N, 13, 13, 3, 1]
            # prob_logits: [N, 13, 13, 3, class_num]
            box_centers, box_sizes, conf_logits, prob_logits = tf.split(
                feature_map, [2, 2, 1, 20], axis=-1)
            #box_centers = tf.nn.sigmoid(box_centers)

            # use some broadcast tricks to get the mesh coordinates
            grid_x = tf.range(grid_size[1], dtype=tf.int32)
            grid_y = tf.range(grid_size[0], dtype=tf.int32)
            grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
            x_offset = tf.reshape(grid_x, (-1, 1))
            y_offset = tf.reshape(grid_y, (-1, 1))
            x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
            # shape: [13, 13, 1, 2]
            x_y_offset = tf.cast(tf.reshape(
                x_y_offset, [grid_size[0], grid_size[1], 1, 2]), tf.float32)

            # get the absolute box coordinates on the feature_map
            box_centers = box_centers + x_y_offset
            # rescale to the original image scale
            box_centers = box_centers * ratio[::-1]

            # avoid getting possible nan value with tf.clip_by_value
            box_sizes = tf.exp(box_sizes) * rescaled_anchors
            # box_sizes = tf.clip_by_value(tf.exp(box_sizes), 1e-9, 100) * rescaled_anchors
            # rescale to the original image scale
            box_sizes = box_sizes * ratio[::-1]

            # shape: [N, 13, 13, 3, 4]
            # last dimension: (center_x, center_y, w, h)
            boxes = tf.concat([box_centers, box_sizes], axis=-1)

            # shape:
            # x_y_offset: [13, 13, 1, 2]
            # boxes: [N, 13, 13, 3, 4], rescaled to the original image scale
            # conf_logits: [N, 13, 13, 3, 1]
            # prob_logits: [N, 13, 13, 3, class_num]
            return x_y_offset, boxes, conf_logits, prob_logits

        def predict(feature_maps):
            '''
            Receive the returned feature_maps from `forward` function,
            the produce the output predictions at the test stage.
            '''
            anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [
                62, 45], [59, 119], [116, 90],  [156, 198],  [373, 326]]
            feature_map_1, feature_map_2, feature_map_3 = feature_maps

            feature_map_anchors = [(feature_map_1, anchors[6:9]),
                                   (feature_map_2, anchors[3:6]),
                                   (feature_map_3, anchors[0:3])]
            reorg_results = [reorg_layer(feature_map, anchors) for (
                feature_map, anchors) in feature_map_anchors]

            def _reshape(result):
                x_y_offset, boxes, conf_logits, prob_logits = result
                grid_size = x_y_offset.get_shape().as_list(
                )[:2]
                boxes = tf.reshape(
                    boxes, [-1, grid_size[0] * grid_size[1] * 3, 4])
                conf_logits = tf.reshape(
                    conf_logits, [-1, grid_size[0] * grid_size[1] * 3, 1])
                prob_logits = tf.reshape(
                    prob_logits, [-1, grid_size[0] * grid_size[1] * 3, 20])
                # shape: (take 416*416 input image and feature_map_1 for example)
                # boxes: [N, 13*13*3, 4]
                # conf_logits: [N, 13*13*3, 1]
                # prob_logits: [N, 13*13*3, class_num]
                return boxes, conf_logits, prob_logits

            boxes_list, confs_list, probs_list = [], [], []
            for result in reorg_results:
                boxes, conf_logits, prob_logits = _reshape(result)
                confs = conf_logits
                probs = prob_logits
                boxes_list.append(boxes)
                confs_list.append(confs)
                probs_list.append(probs)

            # collect results on three scales
            # take 416*416 input image for example:
            # shape: [N, (13*13+26*26+52*52)*3, 4]
            boxes = tf.concat(boxes_list, axis=1)
            # shape: [N, (13*13+26*26+52*52)*3, 1]
            confs = tf.concat(confs_list, axis=1)
            # shape: [N, (13*13+26*26+52*52)*3, class_num]
            probs = tf.concat(probs_list, axis=1)

            center_x, center_y, width, height = tf.split(
                boxes, [1, 1, 1, 1], axis=-1)
            x_min = center_x - width / 2
            y_min = center_y - height / 2
            x_max = center_x + width / 2
            y_max = center_y + height / 2

            boxes = tf.concat([x_min, y_min, x_max, y_max], axis=-1)

            return boxes, confs, probs

        tf.reset_default_graph()
        post_graph = tf.Graph()

        with post_graph.as_default() as g2:
            with g2.name_scope("post-process_graph") as scope:
                feature_maps = [tf.placeholder(dtype=tf.float32, shape=(batch, 13, 13, 3 * (20 + 5))), tf.placeholder(
                    dtype=tf.float32, shape=(batch, 26, 26, 3 * (20 + 5))), tf.placeholder(dtype=tf.float32, shape=(batch, 52, 52, 3 * (20 + 5)))]

                boxes, confs, probs = predict(feature_maps)

                init_op = tf.initialize_all_variables()

        with tf.Session(graph=post_graph) as sess:
            sess.run(init_op)
            while True:
                inp_batch = in_q.get()
                if inp_batch == 'STOP':
                    out_q.put('STOP')
                    break
                yolo_s, yolo_m, yolo_l = inp_batch
                yolo_s = np.transpose(yolo_s, (0, 2, 3, 1))
                yolo_m = np.transpose(yolo_m, (0, 2, 3, 1))
                yolo_l = np.transpose(yolo_l, (0, 2, 3, 1))

                batch_boxes, batch_confs, batch_probs = sess.run([boxes, confs, probs], feed_dict={
                                                                 feature_maps[0]: yolo_s, feature_maps[1]: yolo_m, feature_maps[2]: yolo_l})
                rc.tick()                                                 
                out_q.put((batch_boxes, batch_confs * batch_probs))

    def _presentation(self, in_q, batch_size, presentation_q, rc):
        names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
                 "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]  # used to label the objects
        color_table = get_color_table(class_num=20)
        counter = 0

        while True:
            inp_batch = in_q.get()
            if inp_batch == 'STOP':
                break

            boxes, probs = inp_batch
            for i in range(batch_size):
                if not presentation_q:  # next frames are residual ones and filled with zeros
                    break
                f_boxes, f_score, f_label = cpu_nms(boxes[[i]], probs[[i]], 20)
                
                # updating the offsets based on the original image
                org_img = presentation_q.get()
                H, W, _ = org_img.shape
                r_h, r_w = 416 / H, 416 / W
                r_f = min (r_h, r_w)
                dest_H, dest_W = int(H * r_f), int(W * r_f)
                offset_H, offset_W = (416 - dest_H) // 2, (416 - dest_W) // 2
                if f_boxes is not None:
                    f_boxes = f_boxes - np.array([[offset_W, offset_H, offset_W, offset_H]])
                    f_boxes = f_boxes / r_f
                    for i in range(len(f_boxes)):
                        x0, y0, x1, y1 = f_boxes[i]
                        plot_one_box(org_img, [x0, y0, x1, y1], label=names[f_label[i]] + ', {:.2f}%'.format(f_score[i] * 100), color=color_table[f_label[i]])

                counter += 1
                cv2.imwrite(path.join(self._args.out_path, 'img_{}.jpg'.format(counter)), org_img)
            rc.tick()        




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOv3 accelerated by FPGA')
    parser.add_argument(
        'mode', help='source of the images. camera/video/path/file')
    parser.add_argument('--batch', type=int,
                        help='batch size to pack', default=16)
    parser.add_argument(
        '--show', help='if set, a window will pop up to show the output', action='store_true')
    parser.add_argument('--img_path', type=str,
                        help='path to the folder containing images')

    parser.add_argument('--out_path', type=str, help="path to store output results")               

    args = parser.parse_args()

    y_i = YoloV3Infer(args, q_max_len=5)

    last_thread = y_i.start_threads()

    try:
        while last_thread.is_alive():
            time.sleep(5)
            y_i.print_rates()
    except KeyboardInterrupt:
        y_i.shut_down()
        last_thread.join()