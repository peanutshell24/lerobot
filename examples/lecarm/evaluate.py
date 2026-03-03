#!/usr/bin/env python

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

# 导入LeRobot相关模块
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.robots.lecarm import LecarmClient, LecarmClientConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

# 配置参数
NUM_EPISODES = 2          # 要录制的评估集数
FPS = 30                  # 帧率（每秒帧数）
EPISODE_TIME_SEC = 60     # 每个评估集的持续时间（秒）
TASK_DESCRIPTION = "My task description"  # 任务描述
HF_MODEL_ID = "<hf_username>/<model_repo_id>"  # Hugging Face模型ID
HF_DATASET_ID = "<hf_username>/<eval_dataset_repo_id>"  # Hugging Face数据集ID

# 创建机器人配置和机器人实例
robot_config = LecarmClientConfig(remote_ip="172.18.134.136", id="lekiwi")
robot = LecarmClient(robot_config)

# 从Hugging Face加载预训练策略模型
policy = ACTPolicy.from_pretrained(HF_MODEL_ID)

# 配置数据集特征（动作和观测）
action_features = hw_to_dataset_features(robot.action_features, ACTION)
obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
dataset_features = {**action_features, **obs_features}

# 创建数据集实例
dataset = LeRobotDataset.create(
    repo_id=HF_DATASET_ID,       # Hugging Face数据集ID
    fps=FPS,                     # 帧率
    features=dataset_features,   # 数据集特征
    robot_type=robot.name,       # 机器人类型
    use_videos=True,             # 是否使用视频
    image_writer_threads=4,      # 图像写入线程数
)

# 构建策略处理器（预处理和后处理）
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy,
    pretrained_path=HF_MODEL_ID,
    dataset_stats=dataset.meta.stats,
    # 确保处理器使用与策略相同的设备（CPU/GPU）
    preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
)

# 连接机器人
# 注意：需要先在LeKiwi上运行主机脚本：`python -m lerobot.robots.lecarm.lecarm_host --robot.id=my_awesome_carm`
robot.connect()

# 创建默认处理器（遥操作动作、机器人动作、机器人观测）
teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

# 初始化键盘监听器和可视化工具
listener, events = init_keyboard_listener()  # 用于键盘控制
init_rerun(session_name="lekiwi_evaluate")   # 初始化可视化

# 检查机器人是否已连接
if not robot.is_connected:
    raise ValueError("Robot is not connected!")

print("开始评估循环...")
recorded_episodes = 0  # 已录制集数计数器

# 主循环：录制指定数量的评估集
while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
    log_say(f"运行推理，录制评估集 {recorded_episodes} / {NUM_EPISODES}")

    # 主录制循环
    record_loop(
        robot=robot,                          # 机器人实例
        events=events,                        # 键盘事件
        fps=FPS,                              # 帧率
        policy=policy,                        # 策略模型
        preprocessor=preprocessor,            # 策略前处理器
        postprocessor=postprocessor,          # 策略后处理器
        dataset=dataset,                      # 数据集实例
        control_time_s=EPISODE_TIME_SEC,      # 控制时间（秒）
        single_task=TASK_DESCRIPTION,         # 任务描述
        display_data=True,                    # 是否显示数据
        teleop_action_processor=teleop_action_processor,        # 遥操作处理器
        robot_action_processor=robot_action_processor,          # 机器人动作处理器
        robot_observation_processor=robot_observation_processor, # 机器人观测处理器
    )

    # 如果不是停止录制且需要录制更多集或需要重新录制
    if not events["stop_recording"] and (
        (recorded_episodes < NUM_EPISODES - 1) or events["rerecord_episode"]
    ):
        log_say("重置环境")
        # 运行环境重置循环
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )

    # 如果需要重新录制当前集
    if events["rerecord_episode"]:
        log_say("重新录制当前集")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()  # 清除当前集缓冲区
        continue

    # 保存当前集数据
    dataset.save_episode()
    recorded_episodes += 1  # 增加已录制集数

# 清理资源
log_say("停止录制")
robot.disconnect()  # 断开机器人连接
listener.stop()     # 停止键盘监听
dataset.push_to_hub()  # 将数据集推送到Hugging Face Hub