#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class ImagePublisher:
    def __init__(self):
        
        rospy.init_node('image_publisher_node', anonymous=True)
        self.image_pub = rospy.Publisher('/raw_image', Image, queue_size=10)
        self.bridge = CvBridge()       
        self.cap = cv2.VideoCapture(0)       
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
       
        self.rate = rospy.Rate(10)

    def run(self):
        rospy.loginfo("图像发布节点已启动，正在读取摄像头图像...")
        while not rospy.is_shutdown():
           
            ret, frame = self.cap.read()
            if not ret:
                rospy.logerr("无法读取摄像头图像！")
                break
            
            try:
               
                ros_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
               
                self.image_pub.publish(ros_image)
            except CvBridgeError as e:
                rospy.logerr(f"图像转换失败：{e}")
            
            self.rate.sleep()
        
       
        self.cap.release()
        rospy.loginfo("图像发布节点已停止")

if __name__ == '__main__':
    try:
        img_pub = ImagePublisher()
        img_pub.run()
    except rospy.ROSInterruptException:
        pass
