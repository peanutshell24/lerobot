import serial
import serial.tools.list_ports
import time

BAUD = 115200

# ===================== 扫描串口 =====================
def find_serial_ports():
    ports = list(serial.tools.list_ports.comports())
    result = []

    for p in ports:
        print(f"发现串口: {p.device} | 描述: {p.description}")
        result.append(p.device)

    return result

# ===================== 自动连接 =====================
def auto_connect():
    ports = find_serial_ports()

    if not ports:
        print("❌ 没有找到任何串口设备")
        return None

    for port in ports:
        try:
            print(f"\n尝试连接: {port} ...")
            ser = serial.Serial(port, BAUD, timeout=1)
            time.sleep(2)  # 等ESP32复位

            # 测试读取几行
            for _ in range(5):
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    print(f"✅ {port} 有数据: {line}")
                    return ser

            ser.close()

        except Exception as e:
            print(f"❌ {port} 连接失败: {e}")

    print("❌ 没有找到有效的串口（ESP32可能没在发数据）")
    return None

# ===================== 主程序 =====================
def main():
    ser = auto_connect()

    if ser is None:
        return

    print("\n🎯 开始持续读取数据...\n")

    while True:
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line:
                print(line)
        except Exception as e:
            print("读取错误:", e)
            break

if __name__ == "__main__":
    main()