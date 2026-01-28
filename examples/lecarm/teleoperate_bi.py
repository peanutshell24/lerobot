import argparse
import inspect
import os
import time

from lerobot.robots.lecarm import LecarmClient, LecarmClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.bi_so101_leader import BiSO101Leader, BiSO101LeaderConfig
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# ============ 填写运行参数 ============ #
parser = argparse.ArgumentParser()
parser.add_argument("--use_dummy", action="store_true", help="Do not connect robot, only print actions")
parser.add_argument("--fps", type=int, default=30, help="Main loop frequency (frames per second)")
parser.add_argument("--remote_ip", type=str, default="192.168.1.21", help="Leacrm host IP address")

args = parser.parse_args()

USE_DUMMY = args.use_dummy
FPS = args.fps
# ========================================== #

if USE_DUMMY:
    print("🧪 USE_DUMMY mode enabled: robot will not connect, only print actions.")

# 给leader臂配置
robot_config = LecarmClientConfig(remote_ip=args.remote_ip, id="my_lecarm")
bi_cfg = BiSO101LeaderConfig(
    left_arm_port="/dev/lecarm_left",
    right_arm_port="/dev/lecarm_right",
    id="so101_leader_bi",
)
    # left_arm_port="/dev/ttyACM1",
    # right_arm_port="/dev/ttyACM0",


leader = BiSO101Leader(bi_cfg)
keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")
keyboard = KeyboardTeleop(keyboard_config)
robot = LecarmClient(robot_config)

# 仿真控制逻辑
if not USE_DUMMY:
    robot.connect()
else:
    print("🧪 robot.connect() skipped, only printing actions.")

leader.connect()
keyboard.connect()

# 初始化 rerun 界面
init_rerun(session_name="lecarm_teleop")

if not robot.is_connected or not leader.is_connected or not keyboard.is_connected:
    print("⚠️ 警告: 部分设备未连接,继续运行以debug！")

# 主循环
while True:
    t0 = time.perf_counter()

    observation = robot.get_observation() if not USE_DUMMY else {}

    arm_actions = leader.get_action()
    arm_actions = {f"arm_{k}": v for k, v in arm_actions.items()}
    keyboard_keys = keyboard.get_action()
    base_action = robot._from_keyboard_to_base_action(keyboard_keys)

    action = {**arm_actions, **base_action}
    log_rerun_data(observation, action)

    if USE_DUMMY:
        print(f"[USE_DUMMY] action → {action}")
    else:
        robot.send_action(action)
        #print(f"Sent action → {action}")
        print(f"Sent base_action → {base_action}")

    busy_wait(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
