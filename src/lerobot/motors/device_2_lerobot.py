# device_2_lerobot.py
# 使用你现有的 chassis_comm.py，完全不修改！

from base_serial_control import ChassisComm
import time

def main():
    # 连接到虚拟RS485总线
    chassis = ChassisComm('/tmp/ttyV0', baudrate=115200)
    
    print("🤖 [2号] LeRobot启动，连接到RS485总线")
    print("⏳ [2号] 等待1号授予令牌...")
    
    # 等待令牌
    while not chassis.check_and_acquire_token():
        time.sleep(0.1)
    
    # 发送控制命令
    control_cmd = {
        'arm_left_shoulder_pan.pos': 2.39,
        'x.vel': 0.5,
        'y.vel': 0.0
    }
    
    print("🎯 [2号] 开始与底盘通信...")
    success = chassis.send_base_action(control_cmd)
    
    if success:
        print("📤 [2号] 控制命令已发送")
        # 尝试接收状态（在真实环境中底盘会回复）
        for i in range(3):
            state = chassis.receive_base_state()
            if state:
                print(f"📥 [2号] 收到底盘状态: {state}")
                break
            time.sleep(0.1)
    
    # 释放令牌
    print("🔄 [2号] 释放令牌给1号...")
    chassis.release_token_to_master()
    
    print("✅ [2号] 实验完成！")
    chassis.close()

if __name__ == "__main__":
    main()