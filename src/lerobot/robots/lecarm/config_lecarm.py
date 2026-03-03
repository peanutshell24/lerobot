from dataclasses import dataclass, field

from lerobot.cameras.configs import CameraConfig, Cv2Rotation
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from ..config import RobotConfig

#########《《《这一部分代码指定机器人的基础配置类》》》##############（start）
def lecarm_cameras_config() -> dict[str, CameraConfig]: # 定义摄像头的类型还有串口地址
    return {
        #"fWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEront": OpenCVCameraConfig(
        #    index_or_path="/dev/video0", fps=30, width=640, height=480, rotation=Cv2Rotation.ROTATE_180
        #),
        "wrist_left": OpenCVCameraConfig(
            index_or_path="/dev/video0", fps=30, width=640, height=480, rotation=Cv2Rotation.NO_ROTATION
        ),
        "wrist_right": OpenCVCameraConfig(
            index_or_path="/dev/video4", fps=30, width=640, height=480, rotation=Cv2Rotation.NO_ROTATION
        ),
    }


@RobotConfig.register_subclass("lecarm")
@dataclass
class LecarmConfig(RobotConfig):
    left_port:  str = "/dev/lecarm_left"     # 默认串口地址
    right_port: str = "/dev/lecarm_right"    # 默认串口地址

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    max_relative_target: float | dict[str, float] | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=lecarm_cameras_config)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False
#########《《《这一部分代码指定机器人的基础配置类》》》##############（end）

#########《《《这一部分代码指定机器人的一些特殊需求配置，希望可以随时能够修改》》》##############（start）
@dataclass
class LecarmHostConfig:
    # Network Configuration
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    # Duration of the application
    connection_time_s: int = 6000 #太少了，更更新一下操作时间

    # 看门狗：每0.5秒喂一次狗
    watchdog_timeout_ms: int = 500

    # If robot jitters decrease the frequency and monitor cpu load with `top` in cmd
    max_loop_freq_hz: int = 30
#########《《《这一部分代码指定机器人的一些特殊需求配置，希望可以随时能够修改》》》##############（end）

#########《《《这一部分代码是客户端模式的机器人配置类》》》##############（start）
@RobotConfig.register_subclass("lecarm_client")
@dataclass
class LecarmClientConfig(RobotConfig):
    # 网络配置
    remote_ip: str
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    teleop_keys: dict[str, str] = field(
        default_factory=lambda: {
            # Movement
            "forward": "w",
            "backward": "s",
            "left": "a",
            "right": "d",
            "rotate_left": "q",
            "rotate_right": "e",
            # Speed control
            "speed_up": "r",
            "speed_down": "f",
            # quit teleop
            "quit": "o",
            # shaft up and down
            "left_up":"z",
            "left_down":"x",
            "right_up":"c",
            "right_down":"v",
        }
    )

    cameras: dict[str, CameraConfig] = field(default_factory=lecarm_cameras_config)

    polling_timeout_ms: int = 15
    connect_timeout_s: int = 5
#########《《《这一部分代码是客户端模式的机器人配置类》》》##############（end）
