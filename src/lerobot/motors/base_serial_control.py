# chassis_comm.py
import json
import serial
from typing import Dict, Any, Optional

class ChassisComm:
    """
    树莓派A专用：带令牌控制的底盘通信库
    
    设备ID约定：
    - 1号: 树莓派B (ROS2主控)
    - 2号: 树莓派A (本设备)  
    - 3号: 底盘控制器
    
    通信规则：
    - 只有持有令牌(来自1号)时，才能收发3号数据
    - 发送令牌给1号后，不能再收发3号数据
    """
    
    def __init__(self, serial_port='/dev/ttyS0', baudrate=115200):
        """
        初始化串口
        :param serial_port: 串口路径
        :param baudrate: 波特率
        """
        self.ser = serial.Serial(
            port=serial_port,
            baudrate=baudrate,
            timeout=0.01  # 10ms超时，不阻塞
        )
        self.token_message = "req_control"
        self.req_control = "req_control"
        self.send_action_cmd = "cmd_vel"
        self.get_observation_cmd = "req_observation"
        self.get_obs_message = "state"
        self.release_control = "rel_control"
        self.revoke_control = "revoke_control"

        self.has_token = False  # 是否持有令牌
        
        print(f"[Lecarm] 已连接到 {serial_port}")
    
    def _parse_message(self, json_str: str) -> Optional[Dict[str, Any]]:
        """解析JSON消息，返回完整消息字典"""
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"[ROS2] JSON解析错误: {e}")
            return None
    
    def _is_token_message(self, msg: Dict[str, Any]) -> bool:
        """检查是否为令牌消息（来自ROS2的授权）"""
        return (
            msg.get("type") == self.token_message
        )
    
    def check_and_acquire_token(self) -> bool:
        try:
            if self.has_token:
                return True
                
            if self.ser.in_waiting > 0:
                line = self.ser.readline()
                if line:
                    json_str = line.decode().strip()
                    #print(f"[DEBUG] 2号收到原始数据: '{json_str}'")  # 调试输出
                    if json_str:
                        msg = self._parse_message(json_str)
                        if msg:
                            #print(f"[DEBUG] 2号解析消息: {msg}")  # 调试输出
                            if self._is_token_message(msg):
                                self.has_token = True
                                print("[Lercarm] ✅ 已获取令牌，可以与底盘通信")
                                return True
                        else:
                            print(f"[DEBUG] JSON解析失败: {json_str}")
                
            return False
            
        except Exception as e:
            print(f" 检查令牌错误: {e}")
            return False
    
    def send_base_action(self, command_dict: Dict[str, Any]) -> bool:
        """
        发送控制命令到底盘（JSON字典）
        注意：只有持有令牌时才能发送！
        
        :param command_dict: 控制字典
        :return: True=发送成功，False=没有令牌或发送失败
        """
        if not self.has_token:
            print("[Lercarm] ❌ 没有令牌，无法发送控制命令")
            return False
            
        try:
            # 构建完整消息格式
            message = {
                "type": "cmd_vel",
                "data": command_dict
            }
            
            json_str = json.dumps(message) + '\n'
            self.ser.write(json_str.encode())
            print(f"[Lecarm] 📤 已发送控制命令到底盘")
            return True
            
        except Exception as e:
            print(f"[Lecarm] 发送控制命令错误: {e}")
            return False
    
    def receive_base_state(self) -> Optional[Dict[str, Any]]:
        """
        接收底盘状态（JSON字典）
        注意：只有持有令牌时才能接收！
        
        :return: 底盘状态字典 或 None
        """
        if not self.has_token:
            return None  # 没有令牌，直接返回None
            
        try:
            message = {
                "type": "req_observation",
                "data": None
            }
            json_str = json.dumps(message) + '\n'
            self.ser.write(json_str.encode())
            if self.ser.in_waiting > 0:
                line = self.ser.readline()
                if line:
                    json_str = line.decode().strip()
                    if json_str:
                        msg = self._parse_message(json_str)
                        return msg.get("data")
            
            return None
            
        except Exception as e:
            print(f"[Lecarm] 接收底盘状态错误: {e}")
            return None
    
    def release_token_to_master(self) -> bool:
        """
        释放令牌给1号（主控）
        调用后将不能再收发3号的消息
        
        :return: True=释放成功，False=释放失败
        """
        try:
            # 构建令牌释放消息
            token_release_msg = {
                "type": "rel_control",
                "data":None
            }
            
            json_str = json.dumps(token_release_msg) + '\n'
            self.ser.write(json_str.encode())
            
            # 释放令牌后，清除权限
            self.has_token = False
            print("[Lecarm] 🔄 已释放令牌给主控，停止与底盘通信")
            return True
            
        except Exception as e:
            print(f"[ChassisComm] 释放令牌错误: {e}")
            return False
    
    def close(self):
        """关闭串口"""
        if self.ser:
            self.ser.close()
            print("[ChassisComm] 串口已关闭")