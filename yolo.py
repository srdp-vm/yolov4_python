import darknet as dk
import cv2
import numpy as np
import os
import colorsys
import time

class YOLO():
    def __init__(self, configPath = "", weightPath = "",
                 metaPath = "", classPath = ""):
        if not os.path.exists(configPath):
            raise ValueError("Invalid config path `" +
                             os.path.abspath(configPath) + "`")
        if not os.path.exists(weightPath):
            raise ValueError("Invalid weight path `" +
                             os.path.abspath(weightPath) + "`")
        if not os.path.exists(metaPath):
            raise ValueError("Invalid data file path `" +
                             os.path.abspath(metaPath) + "`")
        if not os.path.exists(classPath):
            raise ValueError("Invalid classes file path `" +
                             os.path.abspath(classPath) + "`")
        self.net = dk.load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)
        self.meta = dk.load_meta(metaPath.encode("ascii"))
        self.class_names = YOLO.getClass(classPath)
        self.darknet_image = dk.make_image(dk.network_width(self.net), dk.network_height(self.net), 3)
        self.thresh = 0.5
        self.colors = self.getColors()


    @staticmethod
    def getClass(classPath):
        with open(classPath) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def getColors(self):
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.
        return colors


    def convertBox(self, box, image_size):
        net_size = (dk.network_height(self.net), dk.network_width(self.net))
        #计算在图片缩放时，填充了多大的边距框
        nsize, top, bottom, left, right = YOLO.calScale(image_size, net_size)
        x = (box[0] - left) / nsize[0] * image_size[1]
        y = (box[1] - top) / nsize[1] * image_size[0]
        w = box[2] / nsize[0] * image_size[1]
        h = box[3] / nsize[1] * image_size[0]
        # x, y, w, h = [box[i] / net_size[i % 2] * image_size[(i+1) % 2] for i in range(len(box))]
        top = int(round(y - h / 2))
        bottom = int(round(y + h / 2))
        left = int(round(x - w / 2))
        right = int(round(x + w / 2))
        return top, bottom, left, right


    def drawBox(self, detections, image, showLabel = False):
        thickness = (image.shape[0] + image.shape[1]) // 300
        thickness -= thickness % 2  # 保证为偶数
        for detection in detections:
            pred_class, score, box = detection
            pred_class = pred_class.decode("utf-8")

            top, bottom, left, right = self.convertBox(box, image.shape)
            top = max(0, top)
            left = max(0, left)
            bottom = min(bottom, image.shape[0])
            right = min(right, image.shape[1])

            label = "{} {:.2f}".format(pred_class, score)
            print(label, (left, top), (right, bottom))

            # 在图片中标注检测项目
            image = cv2.rectangle(image, (left, top), (right, bottom),
                                  self.colors[self.class_names.index(pred_class)], thickness)

            if showLabel:
                label_size, baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                if top - label_size[1] >= 0:
                    label_origin = (left - thickness // 2, top - label_size[1] - baseline)
                    label_end = (label_origin[0] + label_size[0], top)
                else:
                    label_origin = (left - thickness // 2, top + 1)
                    label_end = (label_origin[0] + label_size[0], label_origin[1] + label_size[1] + baseline)

                text_origin = (left, label_end[1] - baseline)  # putText文字以左下角为origin
                image = cv2.rectangle(image, label_origin, label_end,
                                      self.colors[self.class_names.index(pred_class)], -1)
                image = cv2.putText(image, label, text_origin,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        return image


    @staticmethod
    def calScale(srcsize, dstsize):
        """不改变图像长宽比，使用填充缩放法，计算四个边应该填充的border尺寸"""
        sh = srcsize[0]
        sw = srcsize[1]
        w, h = dstsize
        scale = min(w / sh, h / sw)
        nw = int(sw * scale)
        nh = int(sh * scale)
        return (nw, nh),  (h - nh) // 2, h - nh - (h - nh) // 2, (w - nw) // 2, w - nw - (w - nw) // 2


    @staticmethod
    def letterbox_image(image, size):
        """不改变图像长宽比，用padding填充，缩放image到size尺寸"""
        if image.shape != size:
            nsize, top, bottom, left, right = YOLO.calScale(image.shape, size)
            image = cv2.resize(image, nsize)
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return image


    def detect(self, image, showLabel = False):
        begin = time.time()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = YOLO.letterbox_image(image_rgb, (dk.network_width(self.net), dk.network_height(self.net)))
        dk.copy_image_from_bytes(self.darknet_image, resized.tobytes())
        detections = dk.detect_image(self.net, self.meta, self.darknet_image, self.thresh)
        image = self.drawBox(detections, image, showLabel)
        end = time.time()
        print("Time cost:{}s".format(end - begin))
        return image

    def count(self, image):
        """
        计数图片中出现的所有类别数目
        返回值：字典 key-类别， value-数量
        """
        items_count = {}
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = YOLO.letterbox_image(img_rgb, (dk.network_width(self.net), dk.network_height(self.net)))
        dk.copy_image_from_bytes(self.darknet_image, img_resized.tobytes())
        detections = dk.detect_image(self.net, self.meta, self.darknet_image, self.thresh)
        for detection in detections:
            pred_class, score, box = detection
            pred_class = pred_class.decode("utf-8")
            count = items_count.get(pred_class, default=0)
            items_count[pred_class] = count
        return items_count