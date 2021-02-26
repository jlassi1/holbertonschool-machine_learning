#!/usr/bin/env python3
"""0. Initialize Yolo """
import tensorflow.keras as K


class Yolo:
    """ Class YOLO v3 """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """initialization of parametres """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line[:-1] for line in f]
        print(len(self.class_names))
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
