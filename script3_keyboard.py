#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
键盘全向移动控制脚本
任务要求3：通过键盘awds控制全向移动，qe控制转向
参考官方键盘控制示例: src/kuavo_sdk/scripts/keyboard_control/
"""

import rospy
from geometry_msgs.msg import Twist
import sys
import termios
import tty
import select
import os


LINEAR_X_STEP = 0.05  
LINEAR_Y_STEP = 0.05  
ANGULAR_STEP = 0.1    
MAX_LINEAR_X = 0.4   
MAX_LINEAR_Y = 0.2   
MAX_ANGULAR = 0.4     

class KeyboardTeleop:
    def __init__(self):
        self.twist = Twist()
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.rate = rospy.Rate(10)
        
        
        self.settings = termios.tcgetattr(sys.stdin)
        
    def get_key(self):
        """获取键盘输入，非阻塞模式"""
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key
    
    def print_usage(self):
        """打印使用说明"""
        print("\n" + "=" * 50)
        print("Kuavo 键盘全向移动控制")
        print("=" * 50)
        print("移动控制:")
        print("  w/s : 前进/后退 (线速度 x)")  
        print("  a/d : 左移/右移 (线速度 y)") 
        print("  q/e : 左转/右转 (角速度)")
        print("\n其他功能:")
        print("  空格: 紧急停止 (所有速度归零)")
        print("  r   : 重置速度")
        print("  p   : 打印当前速度")
        print("  Ctrl+C: 退出程序")
        print("=" * 50)
        print(f"当前速度: x={self.twist.linear.x:.2f}, y={self.twist.linear.y:.2f}, yaw={self.twist.angular.z:.2f}")
        
    def enforce_limits(self):
        """确保速度在安全范围内"""
        self.twist.linear.x = max(-MAX_LINEAR_X, min(MAX_LINEAR_X, self.twist.linear.x))
        self.twist.linear.y = max(-MAX_LINEAR_Y, min(MAX_LINEAR_Y, self.twist.linear.y))
        self.twist.angular.z = max(-MAX_ANGULAR, min(MAX_ANGULAR, self.twist.angular.z))
    
    def run(self):
        """主循环"""
        self.print_usage()
        
        while not rospy.is_shutdown():
            key = self.get_key()
            
          
            if key == 'w':
                self.twist.linear.x += LINEAR_X_STEP
                print(f"前进 +{LINEAR_X_STEP}")
            elif key == 's':
                self.twist.linear.x -= LINEAR_X_STEP
                print(f"后退 -{LINEAR_X_STEP}")
            elif key == 'a':
                self.twist.linear.y += LINEAR_Y_STEP
                print(f"左移 +{LINEAR_Y_STEP}")
            elif key == 'd':
                self.twist.linear.y -= LINEAR_Y_STEP
                print(f"右移 -{LINEAR_Y_STEP}")
            elif key == 'q':
                self.twist.angular.z += ANGULAR_STEP
                print(f"左转 +{ANGULAR_STEP}")
            elif key == 'e':
                self.twist.angular.z -= ANGULAR_STEP
                print(f"右转 -{ANGULAR_STEP}")
            elif key == ' ':
                self.twist = Twist()  
                print("紧急停止 - 所有速度归零")
            elif key == 'r':
                self.twist = Twist()
                print("重置速度")
            elif key == 'p':
                print(f"\n当前速度状态:")
                print(f"  linear.x: {self.twist.linear.x:.2f} m/s")
                print(f"  linear.y: {self.twist.linear.y:.2f} m/s")
                print(f"  angular.z: {self.twist.angular.z:.2f} rad/s")
            elif key == '\x03':  # Ctrl+C
                break
            
           
            self.enforce_limits()
            
           
            self.pub.publish(self.twist)
            
           
            sys.stdout.write(f"\r当前速度 - x:{self.twist.linear.x:+.2f} y:{self.twist.linear.y:+.2f} yaw:{self.twist.angular.z:+.2f} 按p查看详情")
            sys.stdout.flush()
            
            self.rate.sleep()
        
       
        print("\n\n正在停止机器人...")
        self.pub.publish(Twist())
        rospy.sleep(0.5)

if __name__ == "__main__":
    rospy.init_node("keyboard_teleop")
    teleop = KeyboardTeleop()
    
    try:
        teleop.run()
    except Exception as e:
        print(f"错误: {e}")
    finally:
       
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, teleop.settings)
        print("程序退出")
