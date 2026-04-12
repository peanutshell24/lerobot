# ArUco Pose Tracker

这个程序使用普通 USB 摄像头识别 ArUco 码，并把第一次成功识别时的相机位姿设为原点 `(0, 0, 0)`，之后持续输出相对三维坐标和姿态信息。

## 文件

- `aruco_pose_tracker.py`: 主程序
- `camera_params.sample.json`: 相机内参示例
- `requirements.txt`: Python 依赖

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 运行

先准备相机内参文件：

```bash
cp camera_params.sample.json camera_params.json
```

然后把 `camera_params.json` 中的参数替换成你自己的标定结果。没有真实内参时程序也能运行，但坐标只能作为近似值。

示例：

```bash
python3 aruco_pose_tracker.py --camera 0 --marker-length 0.05
```

如果你的摄像头地址不是 `0`，可以改成你自己的设备号或流地址：

```bash
python3 aruco_pose_tracker.py --camera 1 --marker-length 0.05
python3 aruco_pose_tracker.py --camera /dev/video2 --marker-length 0.05
```

## 参数说明

- `--camera`: 摄像头地址，默认 `0`
- `--marker-length`: ArUco 码边长，单位米
- `--marker-id`: 只跟踪指定 ID；不填时默认使用第一次识别到的 ID
- `--aruco-dict`: 码字典，默认 `DICT_4X4_50`
- `--camera-params`: 相机内参 JSON 路径
- `--width`, `--height`, `--fps`: 请求的采集参数

## 操作

- `q`: 退出
- `r`: 把当前位姿重新设为原点

## 坐标与姿态

- 程序输出的是“当前相机相对于起始时刻”的位姿
- 首次检测到目标码时，当前位置就是 `(0, 0, 0)`
- 姿态以欧拉角 `roll / pitch / yaw` 显示，单位是度

## Linux 测试与 Android 可移植性

当前版本适合先在 Linux 上用 USB 摄像头验证。

为了后续迁移到 Android，代码做了两层划分：

- 图像采集与显示：依赖 `cv2.VideoCapture` 和 `cv2.imshow`
- 位姿计算核心：依赖 OpenCV ArUco 检测、`solvePnP` 和基础矩阵运算

迁移到 Android 时，通常只需要替换“采集/显示”部分，保留 ArUco 检测和位姿变换逻辑即可。若后续你要，我可以再继续给你拆成更适合 Android 复用的纯算法模块，或者直接写一个 Android/OpenCV 版本。
