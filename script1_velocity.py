#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础速度控制脚本 - 向/cmd_vel话题发送速度指令
任务要求1：编写脚本控制机器人以指定速度移动
"""

import rospy
from geometry_msgs.msg import Twist
import sys
import signal

def signal_handler(sig, frame):
    """处理Ctrl+C，发送停止指令"""
    print("\n正在停止机器人...")
    pub_cmd_vel.publish(Twist()) 
    rospy.signal_shutdown("用户中断")
    sys.exit(0)

if __name__ == "__main__":
   
    rospy.init_node("velocity_controller", anonymous=True)
    
   
    pub_cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    
    
    signal.signal(signal.SIGINT, signal_handler)
    
   
    rospy.sleep(0.5)
    
    
    linear_x = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
    linear_y = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    angular_z = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    duration = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
    
   
    twist_msg = Twist()
    twist_msg.linear.x = linear_x
    twist_msg.linear.y = linear_y
    twist_msg.linear.z = 0.0 
    twist_msg.angular.x = 0.0
    twist_msg.angular.y = 0.0
    twist_msg.angular.z = angular_z
    
    
    print(f"速度控制脚本启动")
    print(f"线速度: x={linear_x} m/s, y={linear_y} m/s")
    print(f"角速度: {angular_z} rad/s")
    print(f"持续时间: {duration} 秒")
    print("按 Ctrl+C 停止")
    
    
    rate = rospy.Rate(10)  # 10Hz
    start_time = rospy.Time.now()
    
    while not rospy.is_shutdown():
        
        pub_cmd_vel.publish(twist_msg)
        
        elapsed = (rospy.Time.now() - start_time).to_sec()
        if elapsed >= duration:
            print(f"达到指定时间 {duration} 秒，停止机器人")
            pub_cmd_vel.publish(Twist()) 
            break
            
        rate.sleep()
    
    print("脚本执行完毕")
