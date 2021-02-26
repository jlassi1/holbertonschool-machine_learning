#!/usr/bin/env python3
"""0. Initialize Yolo """
import tensorflow.keras as K
import numpy as np


class Yolo:
    """ Class YOLO v3 """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """initialization of parametres """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line[:-1] for line in f]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, z):
        """ sigmoid function"""
        return (1 / (1 + np.exp(-z)))

    def process_outputs(self, outputs, image_size):
        """  """
        boxes = []
        box_confidences = []
        box_class_probs = []
        img_h, img_w = image_size

        for output in outputs:
            boxes.append(output[..., 0:4])
            box_confidences.append(self.sigmoid(output[..., 4]))
            # print(box_confidences)
            box_class_probs.append(self.sigmoid(output[..., 5:]))
            # print(box_class_probs)
        for i, boxs in enumerate(boxes):
            gr_h, gr_w, anchors_boxes, _ = boxs.shape
            cx = np.indices((gr_h, gr_w, anchors_boxes))[1]
            # print(cx)
            cy = np.indices((gr_h, gr_w, anchors_boxes))[0]
            # print(cy)
            t_x = boxs[..., 0]
            t_y = boxs[..., 1]
            t_w = boxs[..., 2]
            t_h = boxs[..., 3]
            p_w = self.anchors[i, :, 0]
            p_h = self.anchors[i, :, 1]
            bx = (self.sigmoid(t_x) + cx) / gr_w
            by = (self.sigmoid(t_y) + cy) / gr_h
            bw = (np.exp(t_w) * p_w) / self.model.input.shape[1].value
            bh = (np.exp(t_h) * p_h) / self.model.input.shape[1].value
            x1 = bx - bw / 2
            y1 = by - bh / 2
            x2 = x1 + bw
            y2 = y1 + bh
            boxs[..., 0] = x1 * img_w
            boxs[..., 1] = y1 * img_h
            boxs[..., 2] = x2 * img_w
            boxs[..., 3] = y2 * img_h
        return boxes, box_confidences, box_class_probs
