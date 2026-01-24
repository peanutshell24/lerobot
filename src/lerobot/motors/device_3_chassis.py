# device_3_chassis.py
import json
import serial
import time

class Device3Chassis:
    def __init__(self):
        # 连接到虚拟RS485总线
        self.ser = serial.Serial('/tmp/ttyV_USB', baudrate=115200, timeout=0.01)
        print("🚛 [3号] 底盘启动，连接到RS485总线")
        
        # 模拟底盘状态
        self.chassis_state = {
            'x.vel': 0.0,
            'y.vel': 0.0,
            'theta.vel': 0.0, 
            'battery': 85
        }
    
    def process_messages(self):
        """处理总线上的消息"""
        try:
            if self.ser.in_waiting > 0:
                line = self.ser.readline()
                if line:
                    msg_str = line.decode().strip()
                    if msg_str:
                        msg = json.loads(msg_str)
                        
                        # 如果消息是发给自己的
                        if msg.get("to") == 3:
                            print(f"⚙️  [3号] 收到控制命令: {msg.get('data', {})}")
                            
                            # 模拟执行后回复状态给发送方
                            state_msg = {
                                "from": 3,
                                "to": msg.get("from"),
                                "type": "state", 
                                "data": self.chassis_state
                            }
                            reply_str = json.dumps(state_msg) + '\n'
                            self.ser.write(reply_str.encode())
                            print(f"📊 [3号] 已发送状态给{msg.get('from')}号")
                            
        except Exception as e:
            pass
    
    def run(self):
        print("🔄 [3号] 监听总线消息...")
        while True:
            self.process_messages()
            time.sleep(0.01)

if __name__ == "__main__":
    chassis = Device3Chassis()
    try:
        chassis.run()
    except KeyboardInterrupt:
        print("\n[3号] 底盘退出")
    finally:
        chassis.ser.close()