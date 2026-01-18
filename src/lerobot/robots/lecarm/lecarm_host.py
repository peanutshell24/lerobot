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
import base64  # 用于Base64编码/解码
import json    # 用于JSON数据处理
import logging # 日志记录
import time    # 时间相关功能

import cv2     # OpenCV计算机视觉库
import zmq     # ZeroMQ消息队列库

# 从本地模块导入配置类和机器人类
from .config_lecarm import LecarmConfig, LecarmHostConfig  # Lecarm配置类
from .lecarm import Lecarm  # Lecarm机器人类

# 定义Lecarm主机类，负责与远程客户端通信和控制机器人
class LecarmHost:
    def __init__(self, config: LecarmHostConfig):
        """初始化Lecarm主机实例"""
        # 创建ZeroMQ上下文
        self.zmq_context = zmq.Context()
        
        # 创建命令接收套接字（PULL模式，用于接收客户端命令）
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)
        # 设置只保留最新消息（丢弃旧消息）
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
        # 绑定到指定端口
        self.zmq_cmd_socket.bind(f"tcp://*:{config.port_zmq_cmd}")
        
        # 创建观测发送套接字（PUSH模式，用于发送机器人状态给客户端）
        self.zmq_observation_socket = self.zmq_context.socket(zmq.PUSH)
        # 设置只保留最新消息（丢弃旧消息）
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)
        # 绑定到指定端口
        self.zmq_observation_socket.bind(f"tcp://*:{config.port_zmq_observations}")
        
        # 保存配置参数
        self.connection_time_s = config.connection_time_s  # 最大连接时间（秒）
        self.watchdog_timeout_ms = config.watchdog_timeout_ms  # 看门狗超时时间（毫秒）
        self.max_loop_freq_hz = config.max_loop_freq_hz  # 最大循环频率（Hz）

    def disconnect(self):
        """断开连接并清理资源"""
        # 关闭观测发送套接字
        self.zmq_observation_socket.close()
        # 关闭命令接收套接字
        self.zmq_cmd_socket.close()
        # 终止ZeroMQ上下文
        self.zmq_context.term()


# 主函数
def main():
    # 配置日志记录
    logging.info("Config uring Lecarm")
    # 创建Lecarm机器人配置对象
    robot_config = LecarmConfig()
    # 创建Lecarm机器人实例
    robot = Lecarm(robot_config)
    
    # 连接机器人
    logging.info("Connecting Lecarm")
    robot.connect()
    
    # 启动主机代理
    logging.info("Starting HostAgent")
    # 创建主机配置对象
    host_config = LecarmHostConfig()
    # 创建主机实例
    host = LecarmHost(host_config)
    
    # 初始化变量
    last_cmd_time = time.time()  # 记录最后收到命令的时间
    watchdog_active = False      # 看门狗激活状态
    
    # 主循环开始
    logging.info("Waiting for commands...")
    try:
        # 记录循环开始时间
        start = time.perf_counter()
        duration = 0  # 当前运行时间
        
        # 主循环：在连接时间内持续运行
        while duration < host.connection_time_s:
            # 记录单次循环开始时间
            loop_start_time = time.time()
            
            try:
                # 尝试非阻塞接收命令（字符串格式）
                msg = host.zmq_cmd_socket.recv_string(zmq.NOBLOCK)
                # 解析JSON格式的命令数据
                data = dict(json.loads(msg))
                # 发送动作给机器人
                _action_sent = robot.send_action(data)
                # 更新最后收到命令的时间
                last_cmd_time = time.time()
                # 重置看门狗状态
                watchdog_active = False
            except zmq.Again:
                # 没有可用命令时的处理
                if not watchdog_active:
                    logging.warning("No command available")
            except Exception as e:
                # 命令接收失败的处理
                logging.error("Message fetching failed: %s", e)
            
            # 获取当前时间
            now = time.time()
            # 检查是否触发看门狗超时
            if (now - last_cmd_time > host.watchdog_timeout_ms / 1000) and not watchdog_active:
                # 打印超时警告
                logging.warning(
                    f"Command not received for more than {host.watchdog_timeout_ms} milliseconds. Stopping the base."
                )
                # 激活看门狗
                watchdog_active = True
                # 停止底盘运动（安全措施）
                robot.stop_base()
            
            # 获取机器人最新观测数据（关节位置和摄像头图像）
            last_observation = robot.get_observation()
            
            # 处理摄像头图像：转换为JPEG并Base64编码
            for cam_key, _ in robot.cameras.items():
                # 使用OpenCV将图像编码为JPEG格式
                ret, buffer = cv2.imencode(
                    ".jpg", last_observation[cam_key], [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                )
                if ret:
                    # 将JPEG图像数据转换为Base64字符串
                    last_observation[cam_key] = base64.b64encode(buffer).decode("utf-8")
                else:
                    # 编码失败时设为空字符串
                    last_observation[cam_key] = ""
            
            # 发送观测数据给远程客户端
            try:
                # 将观测数据转换为JSON字符串并发送
                host.zmq_observation_socket.send_string(json.dumps(last_observation), flags=zmq.NOBLOCK)
            except zmq.Again:
                # 没有客户端连接时的处理
                logging.info("Dropping observation, no client connected")
            
            # 计算本次循环耗时
            elapsed = time.time() - loop_start_time
            # 计算需要休眠的时间（控制循环频率）
            sleep_time = max(1 / host.max_loop_freq_hz - elapsed, 0)
            # 执行休眠
            time.sleep(sleep_time)
            
            # 更新总运行时间
            duration = time.perf_counter() - start
        
        # 循环结束提示
        print("Cycle time reached.")
    
    except KeyboardInterrupt:
        # 处理键盘中断（Ctrl+C）
        print("Keyboard interrupt received. Exiting...")
    finally:
        # 无论是否发生异常， 执行清理操作
        print("Shutting down Lecarm Host.")
        # 断开机器人连接
        robot.disconnect()
        # 断开主机连接
        host.disconnect()
    
    # 程序结束提示
    logging.info("Finished Lecarm cleanly")


# 程序入口
if __name__ == "__main__":
    main()