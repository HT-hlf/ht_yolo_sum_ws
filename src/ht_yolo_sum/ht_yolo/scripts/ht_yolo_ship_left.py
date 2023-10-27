#!/usr/bin/env python
import argparse
# OPENBLAS_CORETYPE=ARMV8 python ht_yolo_ship.py
import torch.backends.cudnn as cudnn

from utils.datasets_left import *
from utils.utils import *

import cv2

import numpy as np
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from ht_msg.msg import Ht
from ht_msg_ship.msg import Ht_detection
from ht_msg_ship.msg import Ht_detection_list
from cv_bridge import CvBridge, CvBridgeError

import time

def detect(save_img=False):

    rospy.init_node('ht_publish_node_left', anonymous=True)
    image_pub = rospy.Publisher('/ht_image_view/ht_image_left', Image, queue_size=1)  #检测结果可视化发布
    info_pub = rospy.Publisher('/ht_detect_result_left', Ht_detection_list, queue_size=1) # 自定义信息类型发布

    weights =  '/home/nvidia/ht_yolo_sum_ws/src/ht_yolo_sum/ht_yolo/weights/yolov5x.pt'
    view_img=False
    imgsz=640
    conf_thres=0.4
    iou_thres=0.5
    classes=None


    agnostic_nms=False


    # Initialize
    device = torch_utils.select_device('')




    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    model.to(device).eval()
    print('Load model success')
    # model.half()  # to FP16


    # Set Dataloader
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams('0', img_size=imgsz)
    print('Load gmsl camera 2 success')



    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    # colors = [[255,0,0],[0,255,0],[0,0,255]]

    half=False

    # Run inference
    # t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    print('test img')
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    print('test img success')
    print('start detect')
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred,conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

        ht_detection_list=[]

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string


                # Write results
                # ht_detection_list=[]
                for *xyxy, conf, cls in det:

                    # print(torch.tensor(xyxy))
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # normalized xywh
                    # print(xywh)
                    # cv2.circle(im0, (int(xywh[0]),int(xywh[1])),1,(255,0,0),4)

                    label = '%s %.2f' % (names[int(cls)], conf)
                    # print(xyxy)
                    # print(((int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))))
                    # print(xyxy2xywh(torch.tensor(xyxy).view(1, 4)))
                    # print((int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])))
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                    detection_result = Ht_detection()
                    detection_result.object_id = int(cls)
                    detection_result.object_class = label
                    detection_result.x = int(xywh[0])
                    detection_result.y = int(xywh[1])
                    detection_result.width = int(xywh[2])
                    detection_result.height = int(xywh[3])
                    ht_detection_list.append(detection_result)


        # Print time (inference + NMS)
        # print('%sDone. (%.3fs)' % (s, t2 - t1))

        #发布话题
        ros_frame = Image()
        header = Header(stamp=rospy.Time.now())
        header.frame_id = "Camera"
        ros_frame.header = header
        # ros_frame.width = 640
        # ros_frame.height = 480
        # ros_frame.step = 640
        ros_frame.width = 1920
        ros_frame.height = 1080
        ros_frame.step = 1920
        ros_frame.encoding = "bgr8"

        ros_frame.data = np.array(im0).tostring()  # 图片格式转换
        image_pub.publish(ros_frame)  # 发布消息

        if len(ht_detection_list)>0:
            detection_result_list = Ht_detection_list()
            detection_result_list.detections = ht_detection_list
            # print(ht_detection_list)

            info_pub.publish(detection_result_list)
        # rospy.sleep(1)

        #不知道有什么用
        # end = time.time()
        # print("cost time:", end - start)  # 看一下每一帧的执行时间，从而确定合适的rate
        # rate = rospy.Rate(25)  # 10hz
        if rospy.is_shutdown():
            raise StopIteration

        # Stream results
        if view_img:
            cv2.imshow(p, im0)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration
        t2 = torch_utils.time_synchronized()
        print('cost time:',str(t2-t1))

def ht_detect():
    with torch.no_grad():
        detect()

ht_detect()