kuavo_rl_1
介绍
该仓库实现用openCV和ros结合识别红色物体以及用ros发布命令控制机器人运动的功能，并附有演示效果
软件架构
catin_ws为实现用openCV和ros结合识别红色物体功能的工作空间
script1_velocity，script2_gait_demo，script3_keyboard脚本实现用ros发布命令控制机器人运动的功能

使用说明
要实现用openCV和ros结合识别红色物体功能，需要运行kuavo_rl_1\catkin_ws\src\cv_ros_demo\scripts下的两个脚本，分别用于发出图像，接受和识别图像。
要使用script1_velocity，script2_gait_demo，script3_keyboard控制机器人的运动，首先将其放置在乐聚的开源仓库kuavo-ros-opensource/src/kuavo_sdk下，然后进入docker仿真环境，打开gazebo仿真器，运行脚本即可
实现视觉功能的命令
cd ~/catkin_ws
catkin_make
source devel/setup.bash
roscore
新开终端
cd ~/catkin_ws
catkin_make
source devel/setup.bash
rosrun cv_ros_demo img_pub.py
新开终端
cd ~/catkin_ws
catkin_make
source devel/setup.bash
rosrun cv_ros_demo img_sub.py
新开终端
rqt_image_view
实现运动功能的命令行（以脚本3为例）
进入仿真环境和仿真器略，另外需要先下载kuavo-ros-opensource仓库。
cd ~/kuavo-ros-opensource
./docker/run.sh
catkin build kuavo_sdk
source devel/setup.zsh
rosrun kuavo_sdk script3_keyboard.py
其余两个与上面的一致。

参与贡献
1.建立本仓库
2.完成上述功能
