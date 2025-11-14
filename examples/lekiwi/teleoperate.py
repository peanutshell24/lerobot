# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time  # 导入时间模块用于计时和控制循环频率

# 从各模块导入所需的类和配置
from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig  # LeKiwi机器人客户端及配置
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig  # 键盘遥操作
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig  # SO101机械臂控制器
from lerobot.utils.robot_utils import busy_wait  # 精确等待工具函数
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data  # 可视化工具

FPS = 30  # 设置控制循环频率为30帧/秒

# 创建机器人配置对象
# remote_ip: 机器人主机的IP地址
# id: 机器人实例的唯一标识符
robot_config = LeKiwiClientConfig(remote_ip="192.168.1.21", id="my_awesome_carm")

# 创建SO101机械臂控制器配置
# port: 机械臂控制器的串口设备路径
# id: 控制器实例的唯一标识符
teleop_arm_config = SO101LeaderConfig(port="/dev/ttyACM0", id="my_awesome_leader_arm")

# 创建键盘遥操作配置
# id: 键盘控制实例的唯一标识符
keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")

# 实例化机器人客户端对象
robot = LeKiwiClient(robot_config)
# 实例化SO101机械臂控制器
leader_arm = SO101Leader(teleop_arm_config)
# 实例化键盘控制器
keyboard = KeyboardTeleop(keyboard_config)

# 连接到各设备
# 注意：运行此脚本前需确保LeKiwi机器人主机上已运行服务脚本：
# `python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi`
robot.connect()        # 连接机器人
leader_arm.connect()   # 连接机械臂控制器
keyboard.connect()     # 连接键盘控制器

# 初始化rerun可视化工具
# session_name: 可视化会话的名称
init_rerun(session_name="lekiwi_teleop")

# 检查所有设备是否成功连接
if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
    # 若有设备未连接则抛出异常
    raise ValueError("Robot or teleop is not connected!")

print("Starting teleop loop...")  # 开始遥操作循环提示
while True:  # 主控制循环
    t0 = time.perf_counter()  # 记录循环开始时间

    # 获取机器人当前状态观测值
    observation = robot.get_observation()

    # 获取遥操作设备输入并转换为动作指令
    # 从机械臂控制器获取动作
    arm_action = leader_arm.get_action()
    # 为机械臂动作添加"arm_"前缀的命名空间
    arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
    
    # 从键盘获取按键输入
    keyboard_keys = keyboard.get_action()
    # 将键盘输入转换为机器人底盘控制指令
    base_action = robot._from_keyboard_to_base_action(keyboard_keys)

    # 合并机械臂和底盘动作指令
    # 如果有底盘动作则合并，否则只使用机械臂动作
    action = {**arm_action, **base_action} if len(base_action) > 0 else arm_action

    # 将组合后的动作指令发送给机器人执行
    _ = robot.send_action(action)

    # 使用rerun记录当前状态和动作用于可视化
    log_rerun_data(observation=observation, action=action)

    # 计算并执行精确等待，维持设定的FPS
    # 计算剩余时间（确保不小于0）
    elapsed = time.perf_counter() - t0
    sleep_time = max(1.0 / FPS - elapsed, 0.0)
    busy_wait(sleep_time)  # 使用忙等待实现精确计时
