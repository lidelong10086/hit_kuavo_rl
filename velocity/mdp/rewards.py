from __future__ import annotations
"""
env: ManagerBasedRLEnv

这是强化学习环境实例本身，你可以把它想象成所有数据和功能的“大总管”。

    它是什么:ManagerBasedRLEnv 是 Isaac Lab 中基于管理器的RL环境类
它不直接包含所有数据,而是通过几个核心的管理器(Manager)来组织数据。

你能从中获取的信息：
    env.scene: 场景管理器。你可以从这里获取场景中所有实体的信息，包括机器人、传感器、地形等
    获取机器人数据:obot: Articulation = env.scene["robot"]。Articulation对象里包含了关节位置(robot.data.joint_pos)、速度(obot.data.joint_vel)、机身状态(robot.data.root_state_w)等
    获取传感器数据:contact_sensor: ContactSensor = env.scene.sensors["contact_sensor"]。传感器对象里包含了接触力、离地时间等。
env.command_manager: 指令管理器。用于获取发给机器人的高层指令，例如目标速度 env.command_manager.get_command("command_name")
env.action_manager: 动作管理器。可以获取当前步和上一步执行的动作，用于计算动作平滑度等奖励
env.extras: 一个字典，你可以在自定义项里往里面添加指标，用于调试或记录日志

"""

"""
asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")

这是用于指定你想要操作哪个实体（如机器人）以及它的哪些部分（如关节、身体部位） 的配置类。

    它是什么:SceneEntityCfg 是一个配置类,用来在场景中定位一个特定的实体(Entity),并可以进一步细化到它的关节或身体部位

。

你能从中获取的信息：

    asset_cfg.name: 最重要的信息。它告诉函数去场景中找哪个实体。默认值是 "robot"，也就是在场景配置中注册的那个机器人。

    asset_cfg.joint_ids: 如果你只想获取机器人的部分关节数据（比如只关心腿部关节），可以通过这个字段指定关节的索引列表。

    asset_cfg.body_ids: 类似 joint_ids,但用于指定身体部位(Body)的索引,比如只获取两只脚的线速度

    。

如何用它获取信息：
python

# 通过 cfg 获取机器人实体
asset: Articulation = env.scene[asset_cfg.name]

# 利用 cfg 中可能指定的 joint_ids 来获取特定关节的位置
# 如果 cfg 里没指定 joint_ids，asset_cfg.joint_ids 就是 None，这行代码会报错
# 所以通常在函数内部会判断，或者 cfg 在定义时就会指定好 joint_ids
# joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids] 

# 更常见的用法是只通过 cfg 获取整个机器人实体，关节的选择通过其他方式（或硬编码）进行
joint_vel = asset.data.joint_vel # 获取所有关节的速度


"""

"""
sensor_cfg: SceneEntityCfg | None = None

这个参数和 asset_cfg 是同一类东西，但通常指向传感器。

    它是什么：同样是 SceneEntityCfg 类型，但它指向的是场景中的传感器实体，比如接触传感器、IMU等。它被标记为可选（| None），默认值是 None，意味着这个函数不一定需要传感器数据

。

你能从中获取的信息：

    sensor_cfg.name: 指定传感器的名称，例如 "contact_sensor"。

    sensor_cfg.body_ids: 对于接触传感器，这个字段尤其重要。它用于指定传感器上的哪些身体部位（例如哪几只脚）是你关心的

    。

如何用它获取信息：
python

# 如果传入了 sensor_cfg，就获取对应的传感器
if sensor_cfg is not None:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 通过 sensor_cfg.body_ids 选择特定身体部位的数据
    # 例如，获取指定几只脚的接触力
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids] 
"""



import math

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers.manager_base import ManagerTermBase
from omni.isaac.lab.sensors import ContactSensor, RayCaster
from omni.isaac.lab.utils.math import quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv, ManagerBasedEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    #创建传感器对象，获取接触对应的sensor_cfg的传感器数据以及信息。冒号后面的是类的解释。
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    #compute_first_contact方法返回的是一个布尔张量，表示每个身体是否在当前时间步与地面接触。[:, sensor_cfg.body_ids]部分选择了特定身体的接触信息。
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    #last_air_time是一个张量，表示每个身体在当前时间步之前的连续空中时间。[:, sensor_cfg.body_ids]部分选择了特定身体的空中时间信息。
    #关于其实现，物理引擎会在每个时间步更新每个身体的空中状态。当一个身体与地面接触时，其空中时间会被重置为0；当一个身体离开地面时，其空中时间会开始累积，直到再次接触地面。因此，last_air_time反映了每个身体在当前时间步之前的连续空中
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    #reward的计算方式是：对于每个身体，如果它在当前时间步与地面接触（first_contact为True），则奖励等于该身体的空中时间（last_air_time）减去一个预设的阈值（threshold）。
    # 如果该身体没有接触地面，则奖励为0（乘以false）。最后，reward是所有身体奖励的总和。
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    #逻辑是这样的：如果命令的线速度（xy轴）小于0.1（即没有明显的移动指令），则reward乘以0，使得奖励为0；
    # 如果命令的线速度大于0.1，则reward保持不变。这确保了只有在有移动指令时，才会根据空中时间给予奖励。
    #norm函数的作用是计算张量在指定维度上的范数，这里是计算命令的线速度（xy轴）的欧几里得范数（L2范数）。如果这个范数大于0.1，说明有明显的移动指令，reward保持不变；如果这个范数小于或等于0.1，说明没有明显的移动指令，reward被乘以0，最终奖励为0。
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward
    
#限制奖励的最大值，避免过大导致训练不稳定，用跳跃刷奖励
def feet_air_time_clip(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg,
    threshold_min: float,
    threshold_max: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]

    air_time = (last_air_time - threshold_min) * first_contact
    #clamp函数将air_time的值限制在threshold_max - threshold_min的范围内，避免奖励过大导致训练不稳定。
    air_time = torch.clamp(air_time, max=threshold_max - threshold_min)
    reward = torch.sum(air_time, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

#单脚站立奖励，鼓励机器人保持单脚站立，提升步态质量和稳定性
def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    #计算凌空时间
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    #计算接触时间
    #有大说法！我们传进来的sensor_cfg是该函数所需要的所有身体部位在目前dt内的环境的所有传感器的传感器的配置。对于本函数就是两只脚
    #而sensor_cfg.body_ids也是一个列表，列表也能截取列表，从刚开始选择与列表id相同的字典的数据！就是contact_sensor是机器人所有部位的数据
    # 我们在其中的current_contact_time和current_air_time列表中从刚开始选择我们需要的两只脚的数据来计算奖励。
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    #布尔值张量，表示每个身体是否与地面接触
    in_contact = contact_time > 0.0
    #选择的函数，in_contact是condition，contact_time是x，air_time是y。
    # 当in_contact为True时，选择contact_time的值；当in_contact为False时，选择air_time的值。这样，无论身体是否接触地面，都能得到一个时间值（接触时间或空中时间）。
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    #single_stance是一个布尔张量，表示每个环境中是否只有一个身体与地面接触。in_constant,将布尔值转为数字，dim=1就是按行求和。一行！
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    #之前的得到的single_stance是n个环境下的的单脚状态。是[n]的张量。in_mode_time是n*2的张量，表示每个环境下两只脚的接触时间或空中时间。
    #我们就得让single_stance的维度和in_mode_time一样，才能进行后续的计算。unsqueeze(-1)在最后一个维度增加一个维度，把single_stance从[n]变成n*2，这样就可以和in_mode_time进行广播运算了。
    #min函数就是在每个环境中选择两只脚中较小的那个时间值作为奖励，因为我们希望鼓励单脚站立，所以奖励应该基于较短的那个时间值（即单脚站立的时间）。最后，reward是一个长度为n的张量，表示每个环境的奖励值。
    #最后的[0]是因为torch.min返回的是一个包含最小值和索引的元组，本函数就是返回2*n的列表我们只需要最小值部分作为奖励。
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    #截取
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    #时间不能太短！
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(
    env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    #net_forces_w_history是一个四维张量，其维度分别是（N，T，B，3），其中N是传感器数量1，T是历史长度，B是每个传感器的身体数量，3是世界坐标系中的力的三个分量。这个张量记录了每个传感器在过去T个时间步内与每个身体接触时的法向接触力。
    #对两只脚在xyz三个方向上的接触力求范数，其在历史上的最大值，如果大于1就认为是接触了
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    #asset是一个Articulation对象，表示机器人本体。通过env.scene[asset_cfg.name]获取。body_lin_vel_w是一个三维张量，其维度分别是（N，B，3），其中N是环境数量，B是每个环境中身体的数量，3是世界坐标系中的线速度的三个分量。这个张量记录了每个环境中每个身体在世界坐标系中的线速度。
    asset : Articulation = env.scene[asset_cfg.name]
    #body_vel是一个三维张量，其维度分别是（N，B，2），其中N是环境数量，B是每个环境中身体的数量两只脚，2是线速度在xy平面上的两个分量。这个张量记录了每个环境中每个身体在xy平面上的线速度。
    body_vel = asset.data.body_lin_vel_w[:, sensor_cfg.body_ids, :2]
    #速度越大，惩罚越大！
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward

#关节功率惩罚
def joint_power_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    #返回的是机器人对象，通过env.scene[asset_cfg.name]获取。Articulation是一个类，表示机器人本体。它包含了机器人的关节信息、身体信息等。
    # 通过这个对象，我们可以访问机器人的状态数据，比如关节位置、关节速度、施加的扭矩等。
    asset: Articulation = env.scene[asset_cfg.name]
    #功率等于扭矩乘以转速，每一个关节都有吗
    #asset.data返回的是ArticulationData类，包括了机器人的状态数据，比如关节位置、关节速度、施加的扭矩等。
    # applied_torque是一个二维张量，其维度分别是（N，J），其中N是环境数量，J是每个环境中关节的数量。关节被简化为一个维度
    # joint_vel是一个二维张量，其维度分别是（N，J），其中N是环境数量，J是每个环境中关节的数量。这个张量记录了每个环境中每个关节的角速度。
    joint_power = (
        asset.data.applied_torque[:, asset_cfg.joint_ids]
        * asset.data.joint_vel[:, asset_cfg.joint_ids]
    )
    #取绝对值的原因是功率可以是正的（关节在做功）也可以是负的（关节在被动地抵抗），我们都希望惩罚它们，所以取绝对值。最后，reward是一个长度为N的张量，表示每个环境的奖励值。
    return torch.sum(torch.abs(joint_power), dim=1)

#动作平滑度奖励，鼓励动作连续平滑，提升步态自然性和稳定性，用类的方式是因为需要在不同时间步之间存储历史动作信息来计算奖励。
#通过类的方式，我们可以在实例中维护一个属性（prev_prev_action）来存储上一个时间步的动作，从而在每次调用时计算当前动作与前两个动作之间的平滑度奖励。这种设计使得代码更清晰，易于维护和扩展。
#函数是没有状态的，类可以有状态
class action_smoothness_l2(ManagerTermBase):
#就这么理解，当前动作和上个动作全都是从env.action_manager获取的，
# prev_prev_action是我们自己在类中维护的一个属性，用来存储上上个动作的信息。每次调用这个reward函数时，我们都会更新prev_prev_action为当前的上个动作，为下一次调用做准备。


    def __init__(
            #SceneEntityCfg("robot")是调用场景中的机器人实体的配置。这个配置对象包含了与机器人相关的信息，比如关节ID、身体ID等。通过传入这个配置，reward函数可以知道要关注机器人的哪些部分，从而计算相应的奖励。
        self, env: ManagerBasedEnv, cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ):
        super().__init__(env, cfg)
        #初始化储存上上个动作
        self.prev_prev_action = None

    def __call__(
        self, env: ManagerBasedEnv, cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ):
        if self.prev_prev_action is None:
            #第一次调用类，上上动作初始化为上个动作，保证第一次调用时不会因为没有历史动作而报错。实际上在动作的列表里，当前动作和上个动作初始化为0张量
            self.prev_prev_action = env.action_manager.prev_action.clone()
        #进行二阶差分计算，当前动作减去上个动作再减去上个动作减去上上个动作，得到当前动作与前两个动作之间的差异。
        # 然后对这个差异进行平方和，得到一个标量值，表示动作的平滑度。最后返回这个值作为奖励。
        action_smoothness_l2 = torch.sum(
            torch.square(
                env.action_manager.action
                - 2 * env.action_manager.prev_action
                + self.prev_prev_action
            ),
            dim=1,
        )
        #更新上上个动作为当前的上个动作，为下一次调用做准备。把当前的上个动作克隆一份，存储在prev_prev_action中。不会随着更新
        self.prev_prev_action = env.action_manager.prev_action.clone()
        return action_smoothness_l2


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    #sensor_cfg用不到，但是为了接口一致性，保留
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    #如果有传感器配置，就使用传感器数据来计算基座高度，否则直接使用机器人根链接的高度。这样可以让奖励函数适应不同的地形条件。
    if sensor_cfg is not None:
        #创建传感器对象，获取对应的传感器数据以及信息。冒号后面的是类的解释。
        sensor: RayCaster = env.scene[sensor_cfg.name]
        #基座高度的计算方式是：机器人根链接的高度减去传感器测量到的地面高度（ray_hits_w[..., 2]表示传感器测量到的地面在世界坐标系中的z坐标）。如果传感器没有测量到地面（即ray_hits_w[..., 2]为NaN），则使用机器人根链接的高度作为基座高度。
        base_height = asset.data.root_pos_w[:, 2] - sensor.data.ray_hits_w[..., 2].mean(
            dim=-1
        )#就是一个数字了
    else:
        base_height = asset.data.root_link_pos_w[:, 2]
    # Replace NaNs with the base_height
    #无论如何计算基座高度，如果结果是NaN（例如传感器没有测量到地面），我们都将其替换为机器人根链接的高度。这样可以确保奖励函数在任何情况下都有一个有效的基座高度值来计算奖励。
    base_height = torch.nan_to_num(
        base_height, nan=target_height, posinf=target_height, neginf=target_height
    )

    # Compute the L2 squared penalty
    #基座高度与目标高度之间的差异越大，惩罚越大。通过平方这个差异，我们可以确保奖励函数对较大的偏差有更强的惩罚效果，从而鼓励机器人保持接近目标高度。
    return torch.square(base_height - target_height)

#速度跟踪奖励，鼓励机器人跟踪线速度和角速度指令，提升控制精度和响应性
def track_lin_vel_xy_yaw_frame_exp(
    env,
    #std是一个超参数,指数核，控制奖励函数的宽度。较小的std会使奖励函数更陡峭，只有当线速度误差非常小时才会得到较高的奖励；较大的std会使奖励函数更平缓，即使线速度误差较大时也能获得一定的奖励。通过调整std的值，可以控制机器人对线速度指令的跟踪精度要求。
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    #因为线速度指令是以机器人坐标系为参考的，所以我们需要将机器人在世界坐标系中的线速度转换到这个坐标系中。
    # 通过使用机器人根链接的四元数（asset.data.root_link_quat_w）和机器人在世界坐标系中的线速度（asset.data.root_com_lin_vel_w[:, :3]）
    # 我们可以计算出机器人在重力对齐坐标系中的线速度（vel_yaw）。然后，我们将线速度误差（env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]）进行平方和，得到一个标量值，表示线速度误差的大小。
    # 最后，通过指数函数将这个误差转换为奖励值，使得误差越小，奖励越高。
    """
    我们的速度指令是基于机器人自己的坐标系的速度,但是我们从scene获得的速度是机器人相对于世界坐标的速度,我们需要将世界坐标的速度转换为机器人坐标的速度。
    就需要、知道机器人目前的姿态，之后转换。其中速度是三维的张量
    """
    # quat_rotate_inverse函数的作用是将世界坐标系中的线速度转换到机器人坐标系中。它使用了机器人的四元数来进行旋转变换。
    # 就这么理解，我们只考虑机器人在xy平面上的线速度，就比如机器人在上坡时，我们输入水平线速度为1，实际上是希望机器人在斜坡上以1的速度前进，而不是在水平面上以1的速度前进。
    # 通过这个函数，我们可以得到机器人在重力对齐坐标系中的线速度（vel_yaw），然后计算线速度误差并转换为奖励值。
    vel_yaw = quat_rotate_inverse(
        yaw_quat(asset.data.root_link_quat_w), asset.data.root_com_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(
        torch.square(
            env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]
        ),
        dim=1,
    )
    #通过指数函数将线速度误差转换为奖励值，使得误差越小，奖励越高。std参数控制了奖励函数的宽度，较小的std会使奖励函数更陡峭，只有当线速度误差非常小时才会得到较高的奖励；较大的std会使奖励函数更平缓，即使线速度误差较大时也能获得一定的奖励。通过调整std的值，可以控制机器人对线速度指令的跟踪精度要求。
    return torch.exp(-lin_vel_error / std**2)

#角速度跟踪奖励，鼓励机器人跟踪角速度指令，提升转向控制精度和响应性
def track_ang_vel_z_world_exp(
    env,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    #三个轴的旋转，我们只关注绕z轴的旋转，也就是yaw角的旋转。通过比较指令中的角速度（env.command_manager.get_command(command_name)[:, 2]）和机器人当前绕z轴的角速度（asset.data.root_com_ang_vel_w[:, 2]），我们可以计算出角速度误差。
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2]
        - asset.data.root_com_ang_vel_w[:, 2]
    )
    return torch.exp(-ang_vel_error / std**2)

#接触力惩罚，鼓励机器人减少不必要的接触力，提升步态质量和稳定性
def contact_forces(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg, violation_max: float = torch.inf) -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    #历史接触力
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # compute the violation
    violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - threshold
    # compute the penalty
    return torch.sum(violation.clip(min=0.0, max=violation_max), dim=1)

#站立不动奖励，鼓励机器人在没有线速度指令时保持站立不动，提升稳定性和节能性
def stand_still_without_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    #关节位置与默认位置的差异，越小奖励越高
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.abs(diff_angle), dim=-1)
    reward *= (
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < 0.1
    )
    return reward
#重力对齐奖励，鼓励机器人在没有线速度指令时保持重力对齐的姿态，提升稳定性和节能性
def gravity_aligned_when_stopping(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    #速度小于0.05认为没有线速度指令，奖励才会生效
    is_zero_cmd = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < 0.05
    asset: Articulation = env.scene[asset_cfg.name]
    
    # static_flag = getattr(gravity_aligned_when_stopping, "printed", False)
    # if not static_flag and env.num_envs > 0:
    #     print("\n============ DEBUG: Robot Properties ============")
        
    #     print("asset.data attributes:")
    #     for attr in dir(asset.data):
    #         if not attr.startswith('_'): 
    #             try:
    #                 value = getattr(asset.data, attr)
    #                 if isinstance(value, torch.Tensor):
    #                     print(f"  - {attr}: Tensor with shape {value.shape}")
    #                 else:
    #                     print(f"  - {attr}: {type(value)}")
    #             except Exception as e:
    #                 print(f"  - {attr}: Error accessing - {e}")
        
    #     print("\nChecking specific properties for COM and orientation:")
        
    #     if hasattr(asset.data, 'root_link_quat_w'):
    #         print(f"  - root_link_quat_w: {asset.data.root_link_quat_w.shape}")
        
    #     if hasattr(asset.data, 'root_com_pos_w'):
    #         print(f"  - root_com_pos_w: {asset.data.root_com_pos_w.shape}")
    #     elif hasattr(asset.data, 'body_com_pos_w'):
    #         print(f"  - body_com_pos_w: {asset.data.body_com_pos_w.shape}")
        
    #     gravity_aligned_when_stopping.printed = True
    #     print("===============================================\n")
    
    # 
    # 获取躯干的姿态四元数
    root_quat = asset.data.root_link_quat_w
    
    # 计算pitch角度
    w, x, y, z = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
    # 计算pitch角度的公式是基于四元数到欧拉角的转换，特别是绕y轴的旋转。这个公式可以从四元数的定义和欧拉角的关系中推导出来。
    pitch = torch.asin(2.0 * (w * y - x * z))
    
    #pitch接近0时奖励最高
    reward = torch.exp(-5.0 * torch.square(pitch))
    
    masked_reward = torch.zeros_like(reward)
    masked_reward[is_zero_cmd] = reward[is_zero_cmd]
    return masked_reward
def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # -------------------------- 适配1.4.1：find_joints返回格式兼容 --------------------------
        # Isaac Lab 1.4.1 find_joints返回list，需取第一个元素，核心逻辑不变
        env.joint_mirror_joints_cache = []
        for joint_pair in mirror_joints:
            left_joint_ids = asset.find_joints(joint_pair[0])
            right_joint_ids = asset.find_joints(joint_pair[1])
            # 确保取到有效joint id（兼容1.4.1返回格式）
            left_joint_id = left_joint_ids[0] if len(left_joint_ids) > 0 else -1
            right_joint_id = right_joint_ids[0] if len(right_joint_ids) > 0 else -1
            env.joint_mirror_joints_cache.append([[left_joint_id], [right_joint_id]])
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # 跳过无效joint id（避免索引错误）
        if joint_pair[0][0] == -1 or joint_pair[1][0] == -1:
            continue
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward
def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward
def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )
def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)
def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    #sensor_cfg用不到，但是为了接口一致性，保留
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    #如果有传感器配置，就使用传感器数据来计算基座高度，否则直接使用机器人根链接的高度。这样可以让奖励函数适应不同的地形条件。
    if sensor_cfg is not None:
        #创建传感器对象，获取对应的传感器数据以及信息。冒号后面的是类的解释。
        sensor: RayCaster = env.scene[sensor_cfg.name]
        #基座高度的计算方式是：机器人根链接的高度减去传感器测量到的地面高度（ray_hits_w[..., 2]表示传感器测量到的地面在世界坐标系中的z坐标）。如果传感器没有测量到地面（即ray_hits_w[..., 2]为NaN），则使用机器人根链接的高度作为基座高度。
        base_height = asset.data.root_pos_w[:, 2] - sensor.data.ray_hits_w[..., 2].mean(
            dim=-1
        )#就是一个数字了
    else:
        base_height = asset.data.root_link_pos_w[:, 2]
    # Replace NaNs with the base_height
    #无论如何计算基座高度，如果结果是NaN（例如传感器没有测量到地面），我们都将其替换为机器人根链接的高度。这样可以确保奖励函数在任何情况下都有一个有效的基座高度值来计算奖励。
    base_height = torch.nan_to_num(
        base_height, nan=target_height, posinf=target_height, neginf=target_height
    )

    # Compute the L2 squared penalty
    #基座高度与目标高度之间的差异越大，惩罚越大。通过平方这个差异，我们可以确保奖励函数对较大的偏差有更强的惩罚效果，从而鼓励机器人保持接近目标高度。
    return torch.square(base_height - target_height)



def joint_angle_limit_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, limited_joint_pairs: list[list[str]]) -> torch.Tensor:
    """
    奖励函数：限制指定关节对的关节角度在 [0, pi/6] 范围内。
    在范围内时，鼓励角度接近 0（奖励 = MAX - 当前角度）；
    超出范围时，惩罚偏离边界的距离。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 初始化奖励张量（每个环境一个标量）
    reward = torch.zeros(env.num_envs, device=env.device)
    
    # 安全范围（弧度）
    MAX_ANGLE = torch.pi / 20 
    MIN_ANGLE = 0.0

    # 遍历所有需要限制的关节对
    for joint_pair in limited_joint_pairs:
        for joint_name in joint_pair:
            # 1. 获取关节索引
            # find_joints 返回 (indices_list, names_list)
            indices = asset.find_joints(joint_name)[0]  # 取索引列表
            if len(indices) == 0:
                continue  # 关节不存在，跳过（实际不应发生）
            joint_idx = indices[0]  # 取第一个匹配的索引

            # 2. 获取关节位置数据：形状为 (num_envs,)
            # asset.data.joint_pos 形状 (num_envs, num_joints)
            joint_pos = asset.data.joint_pos[:, joint_idx]

            # 3. 计算该关节的奖励贡献
            in_range = (joint_pos >= MIN_ANGLE) & (joint_pos <= MAX_ANGLE)
            clamped = torch.clamp(joint_pos, MIN_ANGLE, MAX_ANGLE)
            r = torch.where(
                in_range,
                MAX_ANGLE - joint_pos,                     # 范围内：越接近 0 奖励越大
                -torch.abs(joint_pos - clamped)            # 范围外：惩罚偏离量
            )
            reward += r

    # 4. 归一化：除以所有受限制关节的总数，使奖励值在合理范围
    total_joints = sum(len(pair) for pair in limited_joint_pairs)
    if total_joints > 0:
        reward /= total_joints

    return reward


def joint_symmetry_reward(
    env: ManagerBasedRLEnv,
    symmetric_joint_pairs: list[list[str]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.1  # 对称误差的惩罚系数，越小惩罚越重
) -> torch.Tensor:
    """
    机器人关节对称奖励：惩罚左右对应关节的位置/速度差异
    核心逻辑：左右对应关节的角度/速度越接近，奖励越高
    """
    asset: Articulation = env.scene[asset_cfg.name]
    eps = 1e-6  # 防止除零
    
    # 1. 缓存对称关节ID（只初始化一次，提升性能）
    cache_key = f"_symmetry_cache_{str(symmetric_joint_pairs)}"
    if not hasattr(env, cache_key):
        cache = {"left_ids": [], "right_ids": []}
        for l_joint, r_joint in symmetric_joint_pairs:
            # 获取左右关节ID
            cache["left_ids"].append(asset.find_joints(l_joint)[1][0])
            cache["right_ids"].append(asset.find_joints(r_joint)[1][0])
        setattr(env, cache_key, cache)
    cache = getattr(env, cache_key)

    # 2. 计算对称误差（位置+速度双约束，更稳）
    total_symmetry_error = torch.zeros(env.num_envs, device=env.device)
    for l_id, r_id in zip(cache["left_ids"], cache["right_ids"]):
        # 关节位置误差（核心：左右关节角度差）
        l_pos = asset.data.joint_pos[:, l_id]
        r_pos = asset.data.joint_pos[:, r_id]
        pos_error = torch.square(l_pos - r_pos)
        
        # 关节速度误差（辅助：左右关节速度差）
        l_vel = asset.data.joint_vel[:, l_id]
        r_vel = asset.data.joint_vel[:, r_id]
        vel_error = torch.square(l_vel - r_vel)
        
        # 位置误差权重更高（70%），速度误差辅助（30%）
        total_symmetry_error += 0.7 * pos_error + 0.3 * vel_error

    # 3. 指数奖励：误差越小，奖励越接近1；误差越大，奖励越接近0
    reward = torch.exp(-total_symmetry_error / (std**2 + eps))
    
    # 限制奖励范围在 [0, 1] 之间
    reward = torch.clamp(reward, 0.0, 1.0)
    
    return reward

