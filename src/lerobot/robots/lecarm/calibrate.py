#!/usr/bin/env python3
"""
独立校准脚本 - 不连接树莓派，只执行校准
"""

import os
import sys
from pathlib import Path

# 添加lerobot路径
lerobot_path = Path(__file__).parent / "src"
sys.path.insert(0, str(lerobot_path))

from lerobot.robots.lecarm.lecarm import Lecarm
from lerobot.robots.lecarm.config import LecarmConfig

def calibrate_only():
    """只执行校准，不进行完整连接"""
    print("=== 独立校准模式 ===")
    
    # 创建配置
    config = LecarmConfig(
        id="my_awesome_lecarm",
        left_port="/dev/ttyACM0",  # 根据实际情况调整
        # 不设置right_port，避免连接右臂
    )
    
    # 创建机器人实例
    robot = Lecarm(config)
    
    # 只执行必要的初始化，不连接电机
    print("初始化机器人配置...")
    robot._load_calibration()  # 加载现有校准数据
    
    # 直接调用校准方法
    print("开始校准流程...")
    robot.calibrate()
    
    print("校准完成！")
    return True

if __name__ == "__main__":
    try:
        calibrate_only()
    except Exception as e:
        print(f"校准失败: {e}")
        import traceback
        traceback.print_exc()