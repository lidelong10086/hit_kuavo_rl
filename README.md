# 任务完成情况

我在设计了新的奖励函数以让其实现不同速度的稳定对称走路，以及增强转向的稳定性，并在训练时不断尝试调试参数

读工程文件，理解仿真以及算法的一些关键逻辑的具体实现，在论文里表述我的收获


## 对称性
原模型出现了左部分手臂侧方抬起，左脚跛脚，针对这一不对称情况，我在~/leju_robot_rl/exts/ext_template/ext_template/tasks/locomotion/velocity/mdp里添加了joint_mirror函数，将输入的关节对里的关节关节角度差取绝对值，之后将其加和作为奖励
之后在rou_env_cfg.py配置，加比较大的权重
## 转向
添加了feet_gait函数，强制要求其步型是抬一只脚，落一只脚。防止其转向时拖脚。以及在走路时也能更好的规范其步态！
## 修正
加上上述函数之后，出现了胳膊转动不合乎人的步态。为了进行限制，加入joint_angle_limit_reward函数，限制其在0-30度之间

## 效果
为了人型机器人走路的泛化能力，我进行了不同情况下的机器人走路，转向的测试如视频所示

转向能力见zhuanxing.mp4，

转向加移动的复杂运动见zhuanxiang_yidong.mp4

单纯的缓慢移动见slow_moving.mp4（一米每秒）

快速移动见fast_moving.mp4（两米每秒）

非常快移动在very_fast_moving.mp4（三米每秒）

但实际上我训练的范围比较大

        self.commands.base_velocity.ranges.lin_vel_x = (-5.0, 5.0)

        self.commands.base_velocity.ranges.lin_vel_y = (-5.0, 5.0)

        self.commands.base_velocity.ranges.ang_vel_z = (-5.0, 5.0)
# 提交材料
## 模型文件
在2026-03-05_11-21-46下的model_11150.pt

## 奖励函数文件以及其配置文件
在velocity文件下。

其余训练过程的参数（比如学习率等参数只有微调）没有过大的改动便不进行展示了
## 论文
在论文的word文档里
## 效果视频
具体的介绍在上述已经说明，有fast_moving.mp4,slow_moving.mp4,very_fast_moving.mp4,zhuanxiang.mp4,zhuanxiang_yidong.mp4

这些我都传入issue区了
