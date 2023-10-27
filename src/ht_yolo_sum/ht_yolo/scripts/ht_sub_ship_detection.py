# @Project -> File     ：scripts -> ht_sub_ship_detection         
# @Author   ：HT
# @Date     ：2023/10/14 上午11:40

#!/usr/bin/env python

import rospy
from ht_msg_ship.msg import Ht_detection_list

def detection_results_callback(data):
    # 这个回调函数会在接收到消息时调用
    rospy.loginfo("Received DetectionResults message with %d detections" % len(data.detections))

    for i, detection in enumerate(data.detections):
        rospy.loginfo("Detection %d:" % (i + 1))
        rospy.loginfo("Object ID: %d" % detection.object_id)
        rospy.loginfo("Object Class: %s" % detection.object_class)
        rospy.loginfo("X: %f" % detection.x)
        rospy.loginfo("Y: %f" % detection.y)
        rospy.loginfo("Width: %f" % detection.width)
        rospy.loginfo("Height: %f" % detection.height)

def detection_listener():
    rospy.init_node('ht_detection_listener', anonymous=True)
    rospy.Subscriber("/ht_detect_result", Ht_detection_list, detection_results_callback)
    rospy.spin()

if __name__ == '__main__':
    try:
        detection_listener()
    except rospy.ROSInterruptException:
        pass
