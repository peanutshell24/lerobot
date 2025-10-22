# 版权所有 2024 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（"许可证"）授权；
# 除非符合许可证要求，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按"原样"分发软件，
# 没有任何明示或暗示的保证或条件。
# 有关特定语言的管理权限，请参阅许可证。

# TODO(aliberts, Steven, Pepijn): 考虑使用 gRPC 替代 zmq？

import base64  # 用于 base64 编解码
import json    # 用于 JSON 数据处理
import logging # 日志记录模块
from functools import cached_property  # 缓存属性装饰器
from typing import Any  # 类型注解支持

import cv2     # OpenCV 图像处理库
import numpy as np  # 数值计算库

# 从项目内部导入常量定义
from lerobot.utils.constants import ACTION, OBS_STATE
# 从项目内部导入自定义异常
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

# 从父模块导入机器人基类
from ..robot import Robot
# 导入 LeCARM 客户端配置类
from .config_lecarm import LecarmClientConfig


class LecarmClient(Robot):
    """LeCARM 机器人的客户端实现，通过 ZeroMQ 与远程机器人通信"""
    
    config_class = LecarmClientConfig  # 配置类
    name = "lecarm_client"  # 机器人名称标识

    def __init__(self, config: LecarmClientConfig):
        """初始化 LeCARM 客户端"""
        import zmq  # 导入 ZeroMQ 库（延迟导入）
        
        self._zmq = zmq  # 保存 zmq 模块引用
        super().__init__(config)  # 调用父类初始化
        self.config = config  # 保存配置对象
        self.id = config.id  # 机器人ID
        self.robot_type = config.type  # 机器人类型

        # 网络配置
        self.remote_ip = config.remote_ip  # 远程机器人IP地址
        self.port_zmq_cmd = config.port_zmq_cmd  # 命令端口
        self.port_zmq_observations = config.port_zmq_observations  # 观测数据端口

        # 遥操作键盘映射配置
        self.teleop_keys = config.teleop_keys

        # 超时设置
        self.polling_timeout_ms = config.polling_timeout_ms  # 轮询超时（毫秒）
        self.connect_timeout_s = config.connect_timeout_s  # 连接超时（秒）

        # ZeroMQ 相关变量
        self.zmq_context = None  # ZeroMQ 上下文
        self.zmq_cmd_socket = None  # 命令发送套接字
        self.zmq_observation_socket = None  # 观测接收套接字

        # 状态缓存
        self.last_frames = {}  # 存储最后一帧图像数据
        self.last_remote_state = {}  # 存储最后一个远程状态

        # 速度级别配置（慢/中/快）
        self.speed_levels = [
            {"xy": 0.1, "theta": 30},  # 慢速
            {"xy": 0.2, "theta": 60},  # 中速
            {"xy": 0.3, "theta": 90},  # 快速
        ]
        self.speed_index = 0  # 当前速度级别索引（初始为慢速）

        # 连接状态
        self._is_connected = False
        self.logs = {}  # 日志存储

    @cached_property
    def _state_ft(self) -> dict[str, type]:
        """定义状态特征的类型（浮点数）"""
        return dict.fromkeys(
            (
                "arm_shoulder_pan.pos",  # 肩部平移位置
                "arm_shoulder_lift.pos",  # 肩部提升位置
                "arm_elbow_flex.pos",    # 肘部弯曲位置
                "arm_wrist_flex.pos",    # 手腕弯曲位置
                "arm_wrist_roll.pos",    # 手腕旋转位置
                "arm_gripper.pos",       # 夹爪位置
                "x.vel",                 # X轴速度
                "y.vel",                 # Y轴速度
                "theta.vel",             # 旋转速度
            ),
            float,  # 所有状态值都是浮点数
        )

    @cached_property
    def _state_order(self) -> tuple[str, ...]:
        """状态键的有序元组（用于保持一致性）"""
        return tuple(self._state_ft.keys())

    @cached_property
    def _cameras_ft(self) -> dict[str, tuple[int, int, int]]:
        """相机特征定义（高度，宽度，通道数）"""
        return {name: (cfg.height, cfg.width, 3) for name, cfg in self.config.cameras.items()}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """组合所有观测特征（状态 + 相机）"""
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """动作特征定义（与状态特征相同）"""
        return self._state_ft

    @property
    def is_connected(self) -> bool:
        """检查是否已连接到机器人"""
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        """校准状态（当前未实现）"""
        pass  # 待实现

    def connect(self) -> None:
        """建立与远程机器人的 ZeroMQ 连接"""
        if self._is_connected:
            raise DeviceAlreadyConnectedError(
                "Lecarm Daemon 已连接。请勿重复调用 `robot.connect()`。"
            )

        zmq = self._zmq
        # 创建 ZeroMQ 上下文
        self.zmq_context = zmq.Context()
        
        # 创建命令发送套接字 (PUSH 模式)
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PUSH)
        zmq_cmd_locator = f"tcp://{self.remote_ip}:{self.port_zmq_cmd}"
        self.zmq_cmd_socket.connect(zmq_cmd_locator)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)  # 只保留最新消息
        
        # 创建观测接收套接字 (PULL 模式)
        self.zmq_observation_socket = self.zmq_context.socket(zmq.PULL)
        zmq_observations_locator = f"tcp://{self.remote_ip}:{self.port_zmq_observations}"
        self.zmq_observation_socket.connect(zmq_observations_locator)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)  # 只保留最新消息
        
        # 使用轮询器检查连接是否成功
        poller = zmq.Poller()
        poller.register(self.zmq_observation_socket, zmq.POLLIN)
        socks = dict(poller.poll(self.connect_timeout_s * 1000))
        
        # 验证连接是否建立
        if self.zmq_observation_socket not in socks or socks[self.zmq_observation_socket] != zmq.POLLIN:
            raise DeviceNotConnectedError("等待 LeCARM 主机连接超时。")
        
        self._is_connected = True  # 更新连接状态

    def calibrate(self) -> None:
        """校准机器人（当前未实现）"""
        pass  # 待实现

    def _poll_and_get_latest_message(self) -> str | None:
        """轮询套接字并获取最新消息字符串"""
        zmq = self._zmq
        poller = zmq.Poller()
        poller.register(self.zmq_observation_socket, zmq.POLLIN)

        try:
            # 在超时时间内轮询消息
            socks = dict(poller.poll(self.polling_timeout_ms))
        except zmq.ZMQError as e:
            logging.error(f"ZMQ 轮询错误: {e}")
            return None

        # 检查是否有新数据
        if self.zmq_observation_socket not in socks:
            logging.info("超时时间内无新数据。")
            return None

        # 获取队列中的所有消息（只保留最后一条）
        last_msg = None
        while True:
            try:
                # 非阻塞接收消息
                msg = self.zmq_observation_socket.recv_string(zmq.NOBLOCK)
                last_msg = msg
            except zmq.Again:  # 没有更多消息
                break

        if last_msg is None:
            logging.warning("轮询器指示有数据，但未能检索到消息。")

        return last_msg

    def _parse_observation_json(self, obs_string: str) -> dict[str, Any] | None:
        """解析 JSON 格式的观测数据字符串"""
        try:
            return json.loads(obs_string)  # 解析 JSON
        except json.JSONDecodeError as e:
            logging.error(f"JSON 观测数据解码错误: {e}")
            return None

    def _decode_image_from_b64(self, image_b64: str) -> np.ndarray | None:
        """将 base64 编码的图像解码为 OpenCV 图像"""
        if not image_b64:
            return None
        try:
            # 解码 base64 数据
            jpg_data = base64.b64decode(image_b64)
            # 转换为 numpy 数组
            np_arr = np.frombuffer(jpg_data, dtype=np.uint8)
            # 解码为 OpenCV 图像
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                logging.warning("cv2.imdecode 返回空图像。")
            return frame
        except (TypeError, ValueError) as e:
            logging.error(f"Base64 图像数据解码错误: {e}")
            return None

    def _remote_state_from_obs(
        self, observation: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """从观测数据中提取帧和状态信息"""
        # 提取状态值（缺失值默认为0.0）
        flat_state = {key: observation.get(key, 0.0) for key in self._state_order}
        
        # 将状态值转换为 numpy 数组
        state_vec = np.array([flat_state[key] for key in self._state_order], dtype=np.float32)
        
        # 构建观测字典
        obs_dict: dict[str, Any] = {**flat_state, OBS_STATE: state_vec}
        
        # 解码所有相机图像
        current_frames: dict[str, np.ndarray] = {}
        for cam_name, image_b64 in observation.items():
            if cam_name not in self._cameras_ft:
                continue  # 跳过非相机字段
            frame = self._decode_image_from_b64(image_b64)
            if frame is not None:
                current_frames[cam_name] = frame
                
        return current_frames, obs_dict

    def _get_data(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """
        从视频套接字获取最新的观测数据
        
        尝试在短时间内检索和解码最新消息。
        如果成功，更新并返回新的帧和状态。
        如果没有新数据或解码失败，返回最后已知的值。
        """
        # 1. 从套接字获取最新消息字符串
        latest_message_str = self._poll_and_get_latest_message()
        
        # 2. 如果没有消息，返回缓存数据
        if latest_message_str is None:
            return self.last_frames, self.last_remote_state
        
        # 3. 解析 JSON 消息
        observation = self._parse_observation_json(latest_message_str)
        
        # 4. 如果 JSON 解析失败，返回缓存数据
        if observation is None:
            return self.last_frames, self.last_remote_state
        
        # 5. 处理有效的观测数据
        try:
            new_frames, new_state = self._remote_state_from_obs(observation)
        except Exception as e:
            logging.error(f"处理观测数据出错，返回最后观测值: {e}")
            return self.last_frames, self.last_remote_state
        
        # 更新缓存
        self.last_frames = new_frames
        self.last_remote_state = new_state
        
        return new_frames, new_state

    def get_observation(self) -> dict[str, Any]:
        """
        从远程机器人获取观测数据：
        - 机械臂当前位置
        - 轮速（转换为机体坐标系速度：x, y, theta）
        - 相机帧
        
        通过 ZeroMQ 接收，转换为机体坐标系速度
        """
        if not self._is_connected:
            raise DeviceNotConnectedError("LecarmClient 未连接。请先运行 `robot.connect()`。")
        
        # 获取帧和状态数据
        frames, obs_dict = self._get_data()
        
        # 处理所有配置的相机
        for cam_name, frame in frames.items():
            if frame is None:
                logging.warning("帧为空")
                # 创建黑色占位图像
                frame = np.zeros((640, 480, 3), dtype=np.uint8)
            # 将帧添加到观测字典
            obs_dict[cam_name] = frame
            
        return obs_dict

    def _from_keyboard_to_base_action(self, pressed_keys: np.ndarray):
        """将键盘输入转换为基本动作命令"""
        # 速度控制逻辑
        if self.teleop_keys["speed_up"] in pressed_keys:
            self.speed_index = min(self.speed_index + 1, 2)  # 提高速度级别
        if self.teleop_keys["speed_down"] in pressed_keys:
            self.speed_index = max(self.speed_index - 1, 0)  # 降低速度级别
            
        # 获取当前速度设置
        speed_setting = self.speed_levels[self.speed_index]
        xy_speed = speed_setting["xy"]  # XY 平面速度
        theta_speed = speed_setting["theta"]  # 旋转速度（度/秒）
        
        # 初始化速度命令
        x_cmd = 0.0  # 前进/后退速度 (m/s)
        y_cmd = 0.0  # 横向速度 (m/s)
        theta_cmd = 0.0  # 旋转速度 (deg/s)
        
        # 根据按键设置速度
        if self.teleop_keys["forward"] in pressed_keys:
            x_cmd += xy_speed  # 前进
        if self.teleop_keys["backward"] in pressed_keys:
            x_cmd -= xy_speed  # 后退
        if self.teleop_keys["left"] in pressed_keys:
            y_cmd += xy_speed  # 左移
        if self.teleop_keys["right"] in pressed_keys:
            y_cmd -= xy_speed  # 右移
        if self.teleop_keys["rotate_left"] in pressed_keys:
            theta_cmd += theta_speed  # 左旋
        if self.teleop_keys["rotate_right"] in pressed_keys:
            theta_cmd -= theta_speed  # 右旋
            
        # 返回速度命令字典
        return {
            "x.vel": x_cmd,
            "y.vel": y_cmd,
            "theta.vel": theta_cmd,
        }

    def configure(self):
        """配置机器人（当前未实现）"""
        pass  # 待实现

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        发送动作命令给 LeCARM 机器人
        
        参数:
            action: 包含目标关节位置的动作字典
            
        返回:
            实际发送的动作字典（可能被裁剪）
        """
        if not self._is_connected:
            raise DeviceNotConnectedError(
                "ManipulatorRobot 未连接。请先运行 `robot.connect()`。"
            )
        
        # 将动作转换为 JSON 字符串并发送
        self.zmq_cmd_socket.send_string(json.dumps(action))
        
        # 将动作转换为有序数组（用于记录）
        actions = np.array([action.get(k, 0.0) for k in self._state_order], dtype=np.float32)
        
        # 构建返回的动作字典
        action_sent = {key: actions[i] for i, key in enumerate(self._state_order)}
        action_sent[ACTION] = actions  # 添加完整动作数组
        return action_sent

    def disconnect(self):
        """断开与机器人的连接并清理资源"""
        if not self._is_connected:
            raise DeviceNotConnectedError(
                "LeCARM 未连接。断开连接前请先运行 `robot.connect()`。"
            )
        
        # 关闭套接字和上下文
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()
        self._is_connected = False  # 更新连接状态