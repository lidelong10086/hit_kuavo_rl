#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动步态切换演示脚本
任务要求2：发送不为0的速度自动向前走，发送为0的速度自动停止
Kuavo的/cmd_vel话题内置了自动步态切换功能
"""

import rospy
from geometry_msgs.msg import Twist
import time

def send_velocity(vx, vy, vyaw, duration):
    """发送速度指令并保持指定时间"""
    twist = Twist()
    twist.linear.x = vx
    twist.linear.y = vy
    twist.angular.z = vyaw
    
    if vx == 0 and vy == 0 and vyaw == 0:
        print(f"发送零速度 → 自动切换到 stance 站立状态")
    else:
        print(f"发送速度 (vx={vx}, vy={vy}, vyaw={vyaw}) → 自动切换到 walk 行走状态")
    
  
    pub.publish(twist)
    end_time = time.time() + duration
    while time.time() < end_time and not rospy.is_shutdown():
        pub.publish(twist)
        rospy.sleep(0.1)
    
   

if __name__ == "__main__":
    rospy.init_node("gait_switch_demo")
    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    
    print("=" * 50)
    print("Kuavo 自动步态切换演示")
    print("根据官方文档: /cmd_vel 话题内置自动切换功能")
    print("- 发送非零速度 → 自动切换到 walk 模式")
    print("- 发送全零速度 → 自动切换到 stance 模式")
    print("=" * 50)
    
    rospy.sleep(1.0)  
    
  
    sequence = [
        (0.0, 0.0, 0.0, 2.0),  
        (0.3, 0.0, 0.0, 4.0),   
        (0.0, 0.0, 0.0, 3.0),  
        (0.0, 0.2, 0.0, 3.0),   
        (0.0, 0.0, 0.0, 2.0),  
        (0.2, 0.0, 0.4, 3.0),   
        (0.0, 0.0, 0.0, 2.0),  
    ]
    
    for i, (vx, vy, vyaw, dur) in enumerate(sequence):
        print(f"\n步骤 {i+1}: ", end="")
        send_velocity(vx, vy, vyaw, dur)
    
    
    print("\n演示结束，发送停止指令")
    send_velocity(0.0, 0.0, 0.0, 1.0)
    print("演示完成")
