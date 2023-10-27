#!/usr/bin/env python

import cv2
import numpy as np
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import time

if __name__ == "__main__":
    capture = cv2.VideoCapture(0)  # 定义摄像头
    rospy.init_node('ht_publish_node_right', anonymous=True)  # 定义节点
    image_pub = rospy.Publisher('/ht_image_view/ht_image_raw_right', Image, queue_size=1)  # 定义话题
    rate = rospy.Rate(20)  # 10hz
    while not rospy.is_shutdown():  # Ctrl C正常退出，如果异常退出会报错device busy！
        start = time.time()
        ret, frame = capture.read()
        # print(frame.shape)
        if ret:  # 如果有画面再执行
            # frame = cv2.flip(frame,0)   #垂直镜像操作
            # frame = cv2.flip(frame, 2)  # 水平镜像操作

            ros_frame = Image()
            header = Header(stamp=rospy.Time.now())
            header.frame_id = "Camera"
            ros_frame.header = header
            ros_frame.width = 1920
            ros_frame.height = 1080
            ros_frame.encoding = "bgr8"
            ros_frame.step = 1920
            ros_frame.data = np.array(frame).tostring()  # 图片格式转换
            image_pub.publish(ros_frame)  # 发布消息
            
            # 
            rate.sleep()
            end = time.time()
            # print("cost time:", end - start)  # 看一下每一帧的执行时间，从而确定合适的rate

    capture.release()
    cv2.destroyAllWindows()
    print("quit successfully!")
