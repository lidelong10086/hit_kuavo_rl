#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class ImageSubscriber:
    def __init__(self):
       
        rospy.init_node('image_subscriber_node', anonymous=True)
        self.bridge = CvBridge()      
        self.image_sub = rospy.Subscriber('/raw_image', Image, self.callback)
        self.image_pub = rospy.Publisher('/processed_image', Image, queue_size=10)
        rospy.loginfo("图像订阅与处理节点已启动，等待接收图像...")

    def detect_red_object(self, frame):
        """红色物体检测核心函数，返回处理后的帧"""
       
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  
        lower_red1 = np.array([0, 80, 50])
        upper_red1 = np.array([10, 255, 255])
        
        lower_red2 = np.array([170, 80, 50])
        upper_red2 = np.array([180, 255, 255])
        
       
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        
       
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) 
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
       
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
          
            largest_contour = max(contours, key=cv2.contourArea)
           
            if cv2.contourArea(largest_contour) > 500:
               
                x, y, w, h = cv2.boundingRect(largest_contour)
              
                center_x = x + w / 2
                center_y = y + h / 2
                
               
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, f"Center: ({center_x:.1f}, {center_y:.1f})", 
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
               
                rospy.loginfo(f"检测到红色物体 - 中心坐标：({center_x:.1f}, {center_y:.1f}) | 矩形区域：x={x}, y={y}, 宽={w}, 高={h}")
            else:
                rospy.loginfo("检测到小轮廓，判定为噪点，忽略")
        else:
            rospy.loginfo("未检测到红色物体")
        
        return frame

    def callback(self, data):
        """订阅回调函数：处理接收到的图像"""
        try:
           
            cv_frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"图像转换失败：{e}")
            return
        
        
        processed_frame = self.detect_red_object(cv_frame)
        
        try:
           
            ros_processed_image = self.bridge.cv2_to_imgmsg(processed_frame, "bgr8")
            self.image_pub.publish(ros_processed_image)
        except CvBridgeError as e:
            rospy.logerr(f"处理后图像发布失败：{e}")

    def run(self):
       
        rospy.spin()

if __name__ == '__main__':
    try:
        img_sub = ImageSubscriber()
        img_sub.run()
    except rospy.ROSInterruptException:
        pass
