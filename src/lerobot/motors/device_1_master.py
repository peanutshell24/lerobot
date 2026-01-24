# device_1_master.py
import json
import serial
import time

class Device1Master:
    def __init__(self):
        # 连接到虚拟RS485总线
        self.ser = serial.Serial('/tmp/ttyV0', baudrate=115200, timeout=0.01)
        print("🚀 [1号] 主控启动，连接到RS485总线")
    
    def send_token_to_device2(self):
        """发送令牌给2号"""
        token_msg = {
            "from": 1,
            "to": 2, 
            "type": "token"
        }
        msg_str = json.dumps(token_msg) + '\n'
        self.ser.write(msg_str.encode())
        print("🔑 [1号] 已发送令牌给2号")
    
    def listen_for_token_release(self):
        """监听2号释放令牌"""
        try:
            if self.ser.in_waiting > 0:
                line = self.ser.readline()
                if line:
                    msg_str = line.decode().strip()
                    if msg_str:
                        msg = json.loads(msg_str)
                        if msg.get("from") == 2 and msg.get("to") == 1 and msg.get("type") == "token_release":
                            print("🔄 [1号] 收到2号释放的令牌")
                            return True
        except Exception as e:
            pass
        return False
    
    def run(self):
        input("🎮 [1号] 按回车键授予2号令牌...")
        self.send_token_to_device2()
        
        print("⏳ [1号] 等待2号释放令牌...")
        while True:
            if self.listen_for_token_release():
                print("✅ [1号] 实验完成！")
                break
            time.sleep(0.1)

if __name__ == "__main__":
    master = Device1Master()
    try:
        master.run()
    except KeyboardInterrupt:
        print("\n[1号] 主控退出")
    finally:
        master.ser.close()