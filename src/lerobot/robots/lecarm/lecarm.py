#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

# 导入必要的库和模块
import logging  # 日志记录
import time     # 时间相关功能
from functools import cached_property  # 缓存属性装饰器
from itertools import chain  # 迭代器工具
from typing import Any  # 类型注解

import numpy as np  # 数值计算库

# 从自定义模块导入所需组件
from lerobot.cameras.utils import make_cameras_from_configs  # 摄像头工具
from lerobot.motors import Motor, MotorCalibration, MotorNormMode  # 电机相关
from lerobot.motors.feetech import (
    FeetechMotorsBus,  # Feetech电机总线
    OperatingMode,     # 电机操作模式
)
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError  # 自定义错误类型

from ..robot import Robot  # 机器人基类
from ..utils import ensure_safe_goal_position  # 安全位置检查工具
from .config_lecarm import LecarmConfig  # Lecarm配置类

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


class Lecarm(Robot):
    """
    Lecarm机器人类，包含三轮全向移动底盘和一个机械臂。
    主控臂（leader arm）在本地连接（笔记本电脑），记录其关节位置并转发给远程跟随臂（follower arm）（应用安全限制）。
    同时，使用键盘遥操作生成轮子的原始速度指令。
    """

    config_class = LecarmConfig  # 指定配置类
    name = "lecarm"  # 机器人名称

    def __init__(self, config: LecarmConfig):
        """初始化Lecarm机器人实例"""
        super().__init__(config)  # 调用父类初始化
        self.config = config  # 保存配置对象
        
        # 根据配置决定使用角度模式还是范围模式
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        
        # 创建Feetech电机总线，包含所有电机配置
        self.bus = FeetechMotorsBus(
            port=self.config.port,  # 串口端口
            motors={
                # 第一个机械臂电机
                "arm_right_shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "arm_right_shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "arm_right_elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "arm_right_wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "arm_right_wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "arm_right_gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
                
                # 第二个机械臂电机
                #"arm_left_shoulder_pan": Motor(7, "sts3215", norm_mode_body),
                #"arm_left_shoulder_lift": Motor(8, "sts3215", norm_mode_body),
                #"arm_left_elbow_flex": Motor(9, "sts3215", norm_mode_body),
                #"arm_left_wrist_flex": Motor(10, "sts3215", norm_mode_body),
                #"arm_left_wrist_roll": Motor(11, "sts3215", norm_mode_body),
                #"arm_left_gripper": Motor(12, "sts3215", MotorNormMode.RANGE_0_100),
                
                # 底盘电机 - 3个全向轮
                #"base_left_wheel": Motor(13, "sts3215", MotorNormMode.RANGE_M100_100),
                #"base_back_wheel": Motor(14, "sts3215", MotorNormMode.RANGE_M100_100),
                #"base_right_wheel": Motor(15, "sts3215", MotorNormMode.RANGE_M100_100),
            },
            calibration=self.calibration,  # 校准数据
        )
        
        # 按功能分类电机名称
        self.arm_motors = [motor for motor in self.bus.motors if motor.startswith("arm")]    # 机械臂电机列表
        self.base_motors = [motor for motor in self.bus.motors if motor.startswith("base")]  # 底盘电机列表
        
        # 根据配置创建摄像头对象
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _state_ft(self) -> dict[str, type]:
        """定义状态观测的特征类型字典（关节位置和底盘速度）"""
        return dict.fromkeys(
            (
                "arm_right_shoulder_pan.pos",  # 右边肩部平移关节位置
                "arm_right_shoulder_lift.pos", # 右边肩部抬升关节位置
                "arm_right_elbow_flex.pos",    # 右边肘部弯曲关节位置
                "arm_right_wrist_flex.pos",    # 右边腕部弯曲关节位置
                "arm_right_wrist_roll.pos",    # 右边腕部旋转关节位置
                "arm_right_gripper.pos",       # 右边夹爪位置

                "arm_left_shoulder_pan.pos",  # 左边肩部平移关节位置
                "arm_left_shoulder_lift.pos", # 左边肩部抬升关节位置
                "arm_left_elbow_flex.pos",    # 左边肘部弯曲关节位置
                "arm_left_wrist_flex.pos",    # 左边腕部弯曲关节位置
                "arm_left_wrist_roll.pos",    # 左边腕部旋转关节位置
                "arm_left_gripper.pos",       # 左边夹爪位置

                #"x.vel",                 # X轴速度（前进/后退）
                #"y.vel",                 # Y轴速度（左右平移）
                #"theta.vel",             # 旋转角速度
            ),
            float,  # 所有特征都是浮点数类型
        )

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """定义摄像头观测的特征形状字典（高度、宽度、通道数）"""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """合并所有观测特征（状态+摄像头）"""
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """定义动作特征（与状态特征相同结构）"""
        return self._state_ft

    @property
    def is_connected(self) -> bool:
        """检查机器人是否已连接（电机总线和所有摄像头都连接）"""
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        """连接机器人并可选进行校准"""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # 连接电机总线
        self.bus.connect()
        
        # 如果未校准且需要校准，则执行校准流程
        if not self.is_calibrated and calibrate:
            logger.info(
                "电机中的校准值与校准文件中的值不匹配或未找到校准文件"
            )
            self.calibrate()

        # 连接所有摄像头
        for cam in self.cameras.values():
            cam.connect()

        # 配置电机参数
        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """检查电机是否已校准"""
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        """执行电机校准流程"""
        if self.calibration:
            # 如果已有校准文件，询问用户是否使用它
            user_input = input(
                f"按回车键使用与ID {self.id} 关联的提供的校准文件，或输入'c'并按回车键运行校准: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"将与ID {self.id} 关联的校准文件写入电机")
                self.bus.write_calibration(self.calibration)
                return
                
        logger.info(f"\n运行 {self} 的校准")

        # 获取所有电机名称
        motors = self.arm_motors + self.base_motors

        # 禁用机械臂扭矩以便手动移动
        self.bus.disable_torque(self.arm_motors)
        for name in self.arm_motors:
            self.bus.write("Operating_Mode", name, OperatingMode.POSITION.value)

        # 提示用户将机器人移动到运动范围中间位置
        input("将机器人移动到其运动范围的中间位置并按回车键....")
        homing_offsets = self.bus.set_half_turn_homings(self.arm_motors)

        # 底盘电机不需要归零偏移
        homing_offsets.update(dict.fromkeys(self.base_motors, 0))

        # 分类电机：全旋转电机和未知范围电机
        full_turn_motor = [
            motor for motor in motors if any(keyword in motor for keyword in ["wheel", "wrist"])
        ]
        unknown_range_motors = [motor for motor in motors if motor not in full_turn_motor]

        print(
            f"依次移动所有机械臂关节（除了 '{full_turn_motor}'）通过其整个运动范围。\n记录位置。按回车键停止..."
        )
        # 记录未知范围电机的运动范围
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        
        # 设置全旋转电机的范围（0-4095，对应0-360度）
        for name in full_turn_motor:
            range_mins[name] = 0
            range_maxes[name] = 4095

        # 创建校准数据字典 现在这里是单臂的校准，需要修改成双臂
        self.calibration = {}
        for name, motor in self.bus.motors.items():
            self.calibration[name] = MotorCalibration(
                id=motor.id,                # 电机ID
                drive_mode=0,               # 驱动模式
                homing_offset=homing_offsets[name],  # 归零偏移
                range_min=range_mins[name],  # 最小范围
                range_max=range_maxes[name], # 最大范围
            )

        # 将校准数据写入电机并保存到文件
        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print("校准已保存到", self.calibration_fpath)

    def configure(self):
        """配置电机参数（PID系数、操作模式等）"""
        # 设置机械臂执行器（位置模式）
        # 假设连接时机械臂处于静止位置，可以安全禁用扭矩以运行校准
        self.bus.disable_torque()
        self.bus.configure_motors()
        
        # 配置每个机械臂电机
        for name in self.arm_motors:
            self.bus.write("Operating_Mode", name, OperatingMode.POSITION.value)  # 位置模式
            # 设置P系数为较低值以避免抖动（默认32）
            self.bus.write("P_Coefficient", name, 16)
            # 设置I系数和D系数为默认值0和32
            self.bus.write("I_Coefficient", name, 0)
            self.bus.write("D_Coefficient", name, 32)

        # 配置底盘电机为速度模式
        for name in self.base_motors:
            self.bus.write("Operating_Mode", name, OperatingMode.VELOCITY.value)

        # 启用所有电机扭矩
        self.bus.enable_torque()

    def setup_motors(self) -> None:
        """设置电机ID（用于初始硬件设置）"""
        for motor in chain(reversed(self.arm_motors), reversed(self.base_motors)):
            input(f"仅将控制器板连接到 '{motor}' 电机并按回车键。")
            self.bus.setup_motor(motor)
            print(f"'{motor}' 电机ID设置为 {self.bus.motors[motor].id}")

    @staticmethod
    def _degps_to_raw(degps: float) -> int:
        """将度/秒转换为原始速度值"""
        steps_per_deg = 4096.0 / 360.0  # 每度的步数（12位编码器）
        speed_in_steps = degps * steps_per_deg  # 计算步数/秒
        speed_int = int(round(speed_in_steps))  # 取整
        
        # 将值限制在16位有符号范围内（-32768到32767）
        if speed_int > 0x7FFF:
            speed_int = 0x7FFF  # 32767 -> 最大正值
        elif speed_int < -0x8000:
            speed_int = -0x8000  # -32768 -> 最小负值
            
        return speed_int

    @staticmethod
    def _raw_to_degps(raw_speed: int) -> float:
        """将原始速度值转换为度/秒"""
        steps_per_deg = 4096.0 / 360.0  # 每度的步数
        magnitude = raw_speed  # 原始速度值
        degps = magnitude / steps_per_deg  # 计算度/秒
        return degps

    def _body_to_wheel_raw(
        self,
        x: float,
        y: float,
        theta: float,
        wheel_radius: float = 0.05,
        base_radius: float = 0.125,
        max_raw: int = 3000,
    ) -> dict:
        """
        将机体坐标系速度转换为轮子原始指令
        
        参数:
          x: X轴线性速度 (m/s)
          y: Y轴线性速度 (m/s)
          theta: 旋转角速度 (deg/s)
          wheel_radius: 每个轮的半径 (米)
          base_radius: 从旋转中心到每个轮的距离 (米)
          max_raw: 每个轮允许的最大原始命令 ( ticks)
        
        返回:
          包含轮子原始命令的字典:
             {"base_left_wheel": 值, "base_back_wheel": 值, "base_right_wheel": 值}
        
        注意:
          - 内部将theta_cmd转换为rad/s用于运动学计算
          - 原始命令是从轮子角速度（deg/s）使用_degps_to_raw()计算的
          - 如果任何命令超过max_raw，所有命令按比例缩小
        """
        # 将旋转速度从deg/s转换为rad/s
        theta_rad = theta * (np.pi / 180.0)
        # 创建机体速度向量 [x, y, theta_rad]
        velocity_vector = np.array([x, y, theta_rad])

        # 定义轮子安装角度（带-90°偏移）
        angles = np.radians(np.array([240, 0, 120]) - 90)
        # 构建运动学矩阵：每行将机体速度映射到轮的线速度
        # 第三列（base_radius）考虑了旋转的影响
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

        # 计算每个轮的线速度（m/s）然后是其角速度（rad/s）
        wheel_linear_speeds = m.dot(velocity_vector)
        wheel_angular_speeds = wheel_linear_speeds / wheel_radius

        # 将轮角速度从rad/s转换为deg/s
        wheel_degps = wheel_angular_speeds * (180.0 / np.pi)

        # 缩放处理：确保不超过最大原始值
        steps_per_deg = 4096.0 / 360.0
        raw_floats = [abs(degps) * steps_per_deg for degps in wheel_degps]
        max_raw_computed = max(raw_floats)
        if max_raw_computed > max_raw:
            scale = max_raw / max_raw_computed
            wheel_degps = wheel_degps * scale

        # 将每个轮的角速度（deg/s）转换为原始整数
        wheel_raw = [self._degps_to_raw(deg) for deg in wheel_degps]

        return {
            "base_left_wheel": wheel_raw[0],
            "base_back_wheel": wheel_raw[1],
            "base_right_wheel": wheel_raw[2],
        }

    def _wheel_raw_to_body(
        self,
        left_wheel_speed,
        back_wheel_speed,
        right_wheel_speed,
        wheel_radius: float = 0.05,
        base_radius: float = 0.125,
    ) -> dict[str, Any]:
        """
        将轮子原始命令反馈转换回机体坐标系速度
        
        参数:
          wheel_raw: 包含原始轮命令的向量 ("base_left_wheel", "base_back_wheel", "base_right_wheel")
          wheel_radius: 每个轮的半径 (米)
          base_radius: 从机器人中心到每个轮的距离 (米)
        
        返回:
          包含机体速度的字典 (x.vel, y.vel, theta.vel)，单位均为m/s
        """

        # 将每个原始命令转换回角速度（deg/s）
        wheel_degps = np.array(
            [
                self._raw_to_degps(left_wheel_speed),
                self._raw_to_degps(back_wheel_speed),
                self._raw_to_degps(right_wheel_speed),
            ]
        )

        # 从deg/s转换为rad/s
        wheel_radps = wheel_degps * (np.pi / 180.0)
        # 从角速度计算每个轮的线速度（m/s）
        wheel_linear_speeds = wheel_radps * wheel_radius

        # 定义轮子安装角度（带-90°偏移）
        angles = np.radians(np.array([240, 0, 120]) - 90)
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

        # 解逆运动学：body_velocity = M⁻¹ · wheel_linear_speeds
        m_inv = np.linalg.inv(m)
        velocity_vector = m_inv.dot(wheel_linear_speeds)
        x, y, theta_rad = velocity_vector
        theta = theta_rad * (180.0 / np.pi)
        return {
            "x.vel": x,        # X轴速度 (m/s)
            "y.vel": y,        # Y轴速度 (m/s)
            "theta.vel": theta, # 旋转速度 (deg/s)
        }

    def get_observation(self) -> dict[str, Any]:
        """获取机器人的完整观测数据（关节位置、速度和摄像头图像）"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # 读取机械臂关节位置和底盘轮子速度
        start = time.perf_counter()
        arm_pos = self.bus.sync_read("Present_Position", self.arm_motors)  # 读取当前位置寄存器
        base_wheel_vel = self.bus.sync_read("Present_Velocity", self.base_motors)  # 读取当前速度寄存器

        # 将轮子原始速度转换为机体坐标系速度
        base_vel = self._wheel_raw_to_body(
            base_wheel_vel["base_left_wheel"],
            base_wheel_vel["base_back_wheel"],
            base_wheel_vel["base_right_wheel"],
        )

        # 格式化机械臂状态数据（添加.pos后缀）
        arm_state = {f"{k}.pos": v for k, v in arm_pos.items()}

        # 合并状态数据
        obs_dict = {**arm_state, **base_vel}

        # 记录读取耗时
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # 从摄像头捕获图像
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()  # 异步读取，不阻塞主线程
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """命令Lecarm移动到目标关节配置。
        
        相对动作幅度可能会根据配置参数`max_relative_target`被裁剪。
        在这种情况下，实际发送的动作与原始动作不同。
        因此，此函数始终返回实际发送的动作。
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # 分离机械臂位置指令和底盘速度指令
        arm_goal_pos = {k: v for k, v in action.items() if k.endswith(".pos")}
        base_goal_vel = {k: v for k, v in action.items() if k.endswith(".vel")}

        # 将机体速度转换为轮子原始速度
        base_wheel_goal_vel = self._body_to_wheel_raw(
            base_goal_vel["x.vel"], base_goal_vel["y.vel"], base_goal_vel["theta.vel"]
        )

        # 当目标位置距离当前位置太远时，限制目标位置
        # 注意：由于需要从跟随者读取数据，预计帧率会降低
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position", self.arm_motors)
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in arm_goal_pos.items()}
            arm_safe_goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)
            arm_goal_pos = arm_safe_goal_pos

        # 发送目标位置到执行器
        arm_goal_pos_raw = {k.replace(".pos", ""): v for k, v in arm_goal_pos.items()}
        self.bus.sync_write("Goal_Position", arm_goal_pos_raw)        # 发送位置指令给机械臂
        self.bus.sync_write("Goal_Velocity", base_wheel_goal_vel)    # 发送速度指令给底盘

        return {**arm_goal_pos, **base_goal_vel}  # 返回实际发送的动作

    def stop_base(self):
        """停止底盘运动（急停功能）"""
        self.bus.sync_write("Goal_Velocity", dict.fromkeys(self.base_motors, 0), num_retry=5)
        logger.info("Base motors stopped")

    def disconnect(self):
        """断开机器人连接并清理资源"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.stop_base()  # 先停止底盘
        self.bus.disconnect(self.config.disable_torque_on_disconnect)  # 断开电机连接
        for cam in self.cameras.values():
            cam.disconnect()  # 断开摄像头连接

        logger.info(f"{self} disconnected.")