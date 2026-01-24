#!/bin/bash

# 授权脚本：grant_permissions.sh
# 需要授权的设备列表，请根据您的实际设备名修改（如 /dev/video0, /dev/ttyUSB0 等）
DEVICES=("/dev/ttyACM0" "/dev/ttyACM1" "/dev/ttyACM2" "/dev/ttyACM3" "/dev/video0" "/dev/video2" "/dev/video4" "/dev/ttyUSB0")

echo "正在为以下设备设置读写权限..."
for DEVICE in "${DEVICES[@]}"; do
    # 检查设备是否存在
    if [ -c "$DEVICE" ] || [ -b "$DEVICE" ]; then  
        echo "找到设备 $DEVICE，正在设置权限..."
        sudo chmod 666 "$DEVICE"
        if [ $? -eq 0 ]; then
            echo "✓ 成功设置 $DEVICE 的权限。"
        else
            echo "✗ 设置 $DEVICE 权限失败。"
        fi
    else
        echo "! 设备 $DEVICE 不存在，请检查设备是否已连接。"
    fi
done
echo "权限设置操作完成。"