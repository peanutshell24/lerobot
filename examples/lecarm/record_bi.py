#!/usr/bin/env python3

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.processor import make_default_processors
from lerobot.robots.lecarm.config_lecarm import LecarmClientConfig
from lerobot.robots.lecarm.lecarm_client import LecarmClient
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.bi_so101_leader import BiSO101Leader, BiSO101LeaderConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

from datetime import datetime
import argparse


def main():
    parser = argparse.ArgumentParser(description="🤖高达精神力感应骨架数据收集")
    parser.add_argument("--dataset", type=str, required=True,
                    help="Dataset repo_id, e.g. lee/record_202601162006")
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to record 简称NER 我开玩笑的")
    parser.add_argument("--fps", type=int, default=30, help="全称 Frames per second")
    parser.add_argument("--episode_time", type=int, default=60, help="每一集的时间 (秒)")
    parser.add_argument("--reset_time", type=int, default=10, help="每一集之间复位的时间 (秒)")
    parser.add_argument("--task_description", type=str, default="My task description", help="任务描述")
    parser.add_argument("--remote_ip", type=str, default="192.168.1.21", help="机器人 host IP")
    parser.add_argument("--robot_id", type=str, default="lekiwi_host", help="机器人 ID")
    args = parser.parse_args()

    # === 遥操作部分参数设置 ===
    robot_config = LecarmClientConfig(remote_ip=args.remote_ip, id=args.robot_id)
    leader_arm_config = BiSO101LeaderConfig(
        left_arm_port="/dev/ttyACM1",
        right_arm_port="/dev/ttyACM0",
        id="so101_leader_bi",
    )

    # 创建键盘遥操作配置
    keyboard_config = KeyboardTeleopConfig()

    # 初始化机器人客户端、双臂遥操作臂和键盘遥操作
    robot = LecarmClient(robot_config)
    leader_arm = BiSO101Leader(leader_arm_config)
    keyboard = KeyboardTeleop(keyboard_config)

    # 创建默认的数据处理器：遥操作动作、机器人动作、机器人观测
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # === 数据集设置 ===
    # 将机器人的动作和观测特征转换为数据集特征格式
    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    # 创建数据集
    dataset = LeRobotDataset.create(
        repo_id=args.dataset,
        fps=args.fps,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )
    print(f"数据集已创建,ID: {dataset.repo_id}")

    # === 连接设备 ===
    robot.connect()
    print("🤖机器人连接！开始！")
    leader_arm.connect()
    print("🤖机械臂控制终端连接！开始！")
    keyboard.connect()
    print("🤖位移控制终端连接！开始！")

    listener, events = init_keyboard_listener()
    init_rerun(session_name="lecarm_record")

    if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
        raise ValueError("😭机器人或控制器未连接！")

    print("🥵开始数据集采集循环...")
    recorded_episodes = 0

    while recorded_episodes < args.num_episodes and not events["stop_recording"]:
        log_say(f"Recording episode {recorded_episodes + 1} of {args.num_episodes}")

        # === Main record loop ===
        record_loop(
            robot=robot,
            events=events,
            fps=args.fps,
            dataset=dataset,
            teleop=[leader_arm, keyboard],
            control_time_s=args.episode_time,
            single_task=args.task_description,
            display_data=True,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )

        # === Reset environment ===
        if not events["stop_recording"] and (
            (recorded_episodes < args.num_episodes - 1) or events["rerecord_episode"]
        ):
            log_say("Reset the environment")
            record_loop(
                robot=robot,
                events=events,
                fps=args.fps,
                teleop=[leader_arm, keyboard],
                control_time_s=args.reset_time,
                single_task=args.task_description,
                display_data=True,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            )

        if events["rerecord_episode"]:
            log_say("Re-record episode")
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        dataset.save_episode()
        recorded_episodes += 1

    # === Clean up ===
    log_say("Stop recording")
    robot.disconnect()
    leader_arm.disconnect()
    keyboard.disconnect()
    listener.stop()
    dataset.finalize()
    dataset.push_to_hub()


if __name__ == "__main__":
    main()
