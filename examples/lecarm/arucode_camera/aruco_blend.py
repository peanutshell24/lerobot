import cv2
import numpy as np
import time
import serial
import threading
import matplotlib.pyplot as plt
from collections import deque
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# =========================================================
# 1. 配置参数
# =========================================================

# 标定参数
MARKER_LENGTH = 0.028   # ArUco码实际尺寸（米）
CUBE_SIZE = 0.05        # 方块实际尺寸（米）

# ArUco字典类型
ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_50

# 相机标定文件路径
CALIB_FILE = "camera_charuco.yml"

# 摄像头ID
CAMERA_ID = 0

# 串口配置（ESP32）
SERIAL_PORT = "/dev/ttyUSB1"
SERIAL_BAUD = 115200

# =========================================================
# 2. 显示控制
# =========================================================

SHOW_DETECTED_MARKERS = True
SHOW_CENTER_POINT = True
SHOW_CENTER_AXES = True
SHOW_CENTER_ID = True
SHOW_CENTER_COORD_TEXT = True
SHOW_AVG_TEXT = True
SHOW_TRAJECTORY = True
SHOW_3D_VIEW = True

# =========================================================
# 3. 轨迹参数
# =========================================================

TRAJECTORY_SECONDS = 3.0
TRAJECTORY_THICKNESS = 2
TRAJECTORY_CIRCLE_RADIUS = 3
TRAJECTORY_FADE = True

CENTER_AXIS_LENGTH = 0.02
EMA_ALPHA = 0.25

# =========================================================
# 4. cube / face 定义
# =========================================================
# cube 坐标系：
# +X = front
# +Y = right
# +Z = up

MARKER_TO_FACE = {
    0: "front",
    1: "back",
    2: "left",    # 修正
    3: "right",   # 修正
    4: "top",
}

DEFAULT_FACE_TURNS_CCW = {
    0: 0,
    1: 4,
    2: 4,
    3: 4,
    4: 2,
}

# =========================================================
# 5. IMU共享状态
# =========================================================

latest_imu = {
    "roll": 0.0,
    "pitch": 0.0,
    "yaw": 0.0,
    "pot": 0,
    "stationary": 1,
}
imu_lock = threading.Lock()

# =========================================================
# 6. 串口读取IMU
# =========================================================

def parse_imu_line(line: str):
    # 期望格式:
    # IMU,ts_ms,ax,ay,az,roll,pitch,yaw,pot,stationary
    parts = line.strip().split(",")
    if len(parts) != 10:
        return None
    if parts[0] != "IMU":
        return None

    try:
        return {
            "roll": float(parts[5]),
            "pitch": float(parts[6]),
            "yaw": float(parts[7]),
            "pot": int(parts[8]),
            "stationary": int(parts[9]),
        }
    except Exception:
        return None


def serial_thread():
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.1)
        print(f"[串口] 已连接 {SERIAL_PORT}")
    except Exception as e:
        print(f"[串口] 打开失败: {e}")
        return

    while True:
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            data = parse_imu_line(line)
            if data is not None:
                with imu_lock:
                    latest_imu.update(data)

        except Exception as e:
            print(f"[串口] 读取异常: {e}")
            time.sleep(0.2)

# =========================================================
# 7. 工具函数：标定 / 检测
# =========================================================

def load_calibration(filepath):
    fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"无法打开标定文件: {filepath}")

    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("dist_coeffs").mat()
    fs.release()

    if camera_matrix is None or dist_coeffs is None:
        raise ValueError("标定文件中没有读取到 camera_matrix / dist_coeffs")

    return camera_matrix, dist_coeffs


def create_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)

    if hasattr(cv2.aruco, "DetectorParameters"):
        detector_params = cv2.aruco.DetectorParameters()
    else:
        detector_params = cv2.aruco.DetectorParameters_create()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
        return detector, aruco_dict, detector_params
    else:
        return None, aruco_dict, detector_params


def detect_markers(gray, detector, aruco_dict, detector_params):
    if detector is not None:
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=detector_params
        )
    return corners, ids, rejected

# =========================================================
# 8. 工具函数：旋转矩阵
# =========================================================

def rot_z_90_ccw():
    return np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1],
    ], dtype=np.float64)


def rot_z_k_ccw(k):
    k = k % 4
    R = np.eye(3, dtype=np.float64)
    for _ in range(k):
        R = rot_z_90_ccw() @ R
    return R


def euler_to_rotation_matrix(roll_deg, pitch_deg, yaw_deg):
    rx = np.deg2rad(roll_deg)
    ry = np.deg2rad(pitch_deg)
    rz = np.deg2rad(yaw_deg)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx),  np.cos(rx)],
    ], dtype=np.float64)

    Ry = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)],
    ], dtype=np.float64)

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz),  np.cos(rz), 0],
        [0, 0, 1],
    ], dtype=np.float64)

    return Rz @ Ry @ Rx

# =========================================================
# 9. 工具函数：face基础朝向
# =========================================================

FACE_BASE_ROT = {
    "top": np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], dtype=np.float64),

    "front": np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=np.float64),

    "right": np.array([
        [0, -1, 0],
        [0,  0, 1],
        [1,  0, 0],
    ], dtype=np.float64),

    "back": np.array([
        [ 0,  0, -1],
        [-1,  0,  0],
        [ 0,  1,  0],
    ], dtype=np.float64),

    "left": np.array([
        [1,  0,  0],
        [0,  0, -1],
        [0,  1,  0],
    ], dtype=np.float64),
}

# =========================================================
# 10. marker -> cube 姿态
# =========================================================

def get_marker_to_cube_rotation(marker_id):
    if marker_id not in MARKER_TO_FACE:
        raise KeyError(f"marker_id {marker_id} 未定义 face")

    face = MARKER_TO_FACE[marker_id]
    R_base = FACE_BASE_ROT[face]

    turns = DEFAULT_FACE_TURNS_CCW.get(marker_id, 0)
    R_turn = rot_z_k_ccw(turns)

    R_cube_from_marker = R_base @ R_turn
    return R_cube_from_marker


def marker_pose_to_cube_pose(rvec, tvec, cube_size, marker_id):
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)

    R_marker_to_cam, _ = cv2.Rodrigues(rvec)
    R_cube_from_marker = get_marker_to_cube_rotation(marker_id)
    R_marker_from_cube = R_cube_from_marker.T

    face = MARKER_TO_FACE[marker_id]

    if face == "top":
        center_in_cube = np.array([0.0, 0.0, -cube_size / 2.0], dtype=np.float64)
    elif face == "front":
        center_in_cube = np.array([-cube_size / 2.0, 0.0, 0.0], dtype=np.float64)
    elif face == "right":
        center_in_cube = np.array([0.0, -cube_size / 2.0, 0.0], dtype=np.float64)
    elif face == "back":
        center_in_cube = np.array([cube_size / 2.0, 0.0, 0.0], dtype=np.float64)
    elif face == "left":
        center_in_cube = np.array([0.0, cube_size / 2.0, 0.0], dtype=np.float64)
    else:
        raise ValueError(f"未知 face: {face}")

    center_in_marker = R_marker_from_cube @ center_in_cube.reshape(3, 1)
    center_cam = R_marker_to_cam @ center_in_marker + tvec

    return center_cam.reshape(3)

# =========================================================
# 11. 投影 / 绘图
# =========================================================

def project_cam_point(point_cam, camera_matrix, dist_coeffs):
    point_cam = np.asarray(point_cam, dtype=np.float32).reshape(1, 1, 3)
    rvec = np.zeros((3, 1), dtype=np.float32)
    tvec = np.zeros((3, 1), dtype=np.float32)
    img_pts, _ = cv2.projectPoints(point_cam, rvec, tvec, camera_matrix, dist_coeffs)
    return tuple(np.round(img_pts[0, 0]).astype(int))


def draw_text(img, text, org, color=(0, 255, 0), scale=0.6, thickness=2):
    cv2.putText(
        img, text, org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, thickness, cv2.LINE_AA
    )


def draw_center_axes_from_pose(display, center_cam, R_draw, axis_length,
                               camera_matrix, dist_coeffs):
    center_cam = np.asarray(center_cam, dtype=np.float64).reshape(3)
    R_draw = np.asarray(R_draw, dtype=np.float64).reshape(3, 3)

    x_dir = R_draw[:, 0]
    y_dir = R_draw[:, 1]
    z_dir = R_draw[:, 2]

    p0_cam = center_cam
    px_cam = center_cam + axis_length * x_dir
    py_cam = center_cam + axis_length * y_dir
    pz_cam = center_cam + axis_length * z_dir

    p0 = project_cam_point(p0_cam, camera_matrix, dist_coeffs)
    px = project_cam_point(px_cam, camera_matrix, dist_coeffs)
    py = project_cam_point(py_cam, camera_matrix, dist_coeffs)
    pz = project_cam_point(pz_cam, camera_matrix, dist_coeffs)

    cv2.line(display, p0, px, (0, 0, 255), 2)   # X 红
    cv2.line(display, p0, py, (0, 255, 0), 2)   # Y 绿
    cv2.line(display, p0, pz, (255, 0, 0), 2)   # Z 蓝

    draw_text(display, "X", (px[0] + 4, px[1] + 4), (0, 0, 255), 0.5, 1)
    draw_text(display, "Y", (py[0] + 4, py[1] + 4), (0, 255, 0), 0.5, 1)
    draw_text(display, "Z", (pz[0] + 4, pz[1] + 4), (255, 0, 0), 0.5, 1)

# =========================================================
# 12. 平滑 / 轨迹
# =========================================================

def ema_filter(prev_pt, new_pt, alpha=0.25):
    if prev_pt is None:
        return np.asarray(new_pt, dtype=np.float64)
    prev_pt = np.asarray(prev_pt, dtype=np.float64)
    new_pt = np.asarray(new_pt, dtype=np.float64)
    return alpha * new_pt + (1.0 - alpha) * prev_pt


def update_trajectory(trajectory, point_px, now_ts, keep_seconds=3.0):
    if point_px is not None:
        trajectory.append((now_ts, point_px))

    while trajectory and (now_ts - trajectory[0][0] > keep_seconds):
        trajectory.popleft()


def draw_trajectory(display, trajectory, now_ts):
    if len(trajectory) < 1:
        return

    pts = [p for _, p in trajectory]
    ts_list = [t for t, _ in trajectory]

    for i, p in enumerate(pts):
        if TRAJECTORY_FADE:
            age = now_ts - ts_list[i]
            ratio = max(0.0, min(1.0, 1.0 - age / TRAJECTORY_SECONDS))
            intensity = int(255 * ratio)
            color = (intensity, 0, intensity)
        else:
            color = (255, 0, 255)

        cv2.circle(display, p, TRAJECTORY_CIRCLE_RADIUS, color, -1)

    for i in range(1, len(pts)):
        if TRAJECTORY_FADE:
            age = now_ts - ts_list[i]
            ratio = max(0.0, min(1.0, 1.0 - age / TRAJECTORY_SECONDS))
            intensity = int(255 * ratio)
            color = (intensity, 0, intensity)
        else:
            color = (255, 0, 255)

        cv2.line(display, pts[i - 1], pts[i], color, TRAJECTORY_THICKNESS)

# =========================================================
# 13. 3D显示
# =========================================================

def make_cube_vertices(size=0.05):
    s = size / 2.0
    return np.array([
        [-s, -s, -s],
        [ s, -s, -s],
        [ s,  s, -s],
        [-s,  s, -s],
        [-s, -s,  s],
        [ s, -s,  s],
        [ s,  s,  s],
        [-s,  s,  s],
    ], dtype=np.float64)


def transform_points(points, R, t):
    return (R @ points.T).T + t.reshape(1, 3)


def draw_cube_3d(ax, R, t, size=0.05, color='cyan', alpha=0.45):
    verts = make_cube_vertices(size)
    verts_w = transform_points(verts, R, t)

    faces = [
        [verts_w[0], verts_w[1], verts_w[2], verts_w[3]],
        [verts_w[4], verts_w[5], verts_w[6], verts_w[7]],
        [verts_w[0], verts_w[1], verts_w[5], verts_w[4]],
        [verts_w[2], verts_w[3], verts_w[7], verts_w[6]],
        [verts_w[1], verts_w[2], verts_w[6], verts_w[5]],
        [verts_w[4], verts_w[7], verts_w[3], verts_w[0]],
    ]

    poly = Poly3DCollection(faces, facecolors=color, edgecolors='k', alpha=alpha)
    ax.add_collection3d(poly)


def draw_axes_3d(ax, R, t, axis_len=0.03):
    o = t.reshape(3)
    x1 = o + R[:, 0] * axis_len
    y1 = o + R[:, 1] * axis_len
    z1 = o + R[:, 2] * axis_len

    ax.plot([o[0], x1[0]], [o[1], x1[1]], [o[2], x1[2]], 'r-', linewidth=2)
    ax.plot([o[0], y1[0]], [o[1], y1[1]], [o[2], y1[2]], 'g-', linewidth=2)
    ax.plot([o[0], z1[0]], [o[1], z1[1]], [o[2], z1[2]], 'b-', linewidth=2)


def setup_3d_axes(ax, center=None):
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])

    if center is None:
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(-0.2, 0.2)
        ax.set_zlim(0.0, 0.6)
    else:
        cx, cy, cz = center
        r = max(0.12, CUBE_SIZE * 2.0)
        ax.set_xlim(cx - r, cx + r)
        ax.set_ylim(cy - r, cy + r)
        ax.set_zlim(max(0.0, cz - r), cz + r)

    ax.view_init(elev=25, azim=-60)

# =========================================================
# 14. 主程序
# =========================================================

def main():
    camera_matrix, dist_coeffs = load_calibration(CALIB_FILE)
    print("[INFO] 已加载相机内参")
    print(camera_matrix)
    print(dist_coeffs)

    detector, aruco_dict, detector_params = create_detector()

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头")
        return

    threading.Thread(target=serial_thread, daemon=True).start()

    trajectory = deque()
    smoothed_center_px = None

    if SHOW_3D_VIEW:
        plt.ion()
        fig = plt.figure("Cube 3D Pose")
        ax3d = fig.add_subplot(111, projection='3d')
    else:
        ax3d = None

    print("按 q 或 ESC 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] 读取摄像头失败")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display = frame.copy()

        corners, ids, _ = detect_markers(gray, detector, aruco_dict, detector_params)

        cube_centers_cam = []
        center_pixels = []
        now_ts = time.time()

        with imu_lock:
            imu = latest_imu.copy()

        if ids is None:
            print("\r[调试] 当前未检测到 marker                         ", end="")
        else:
            print(f"\r[调试] 检测到 marker IDs: {ids.flatten().tolist()}                     ", end="")

        if ids is not None and len(ids) > 0:
            if SHOW_DETECTED_MARKERS:
                cv2.aruco.drawDetectedMarkers(display, corners, ids)

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, MARKER_LENGTH, camera_matrix, dist_coeffs
            )

            for i in range(len(ids)):
                marker_id = int(ids[i][0])
                rvec = rvecs[i]
                tvec = tvecs[i]

                if marker_id not in MARKER_TO_FACE:
                    continue

                try:
                    center_cam = marker_pose_to_cube_pose(
                        rvec, tvec, CUBE_SIZE, marker_id
                    )
                except Exception as e:
                    print(f"\n[WARN] marker {marker_id} 计算失败: {e}")
                    continue

                cube_centers_cam.append(center_cam)

                center_px = project_cam_point(center_cam, camera_matrix, dist_coeffs)
                center_pixels.append(center_px)

                if SHOW_CENTER_ID:
                    draw_text(
                        display,
                        f"ID {marker_id}",
                        (center_px[0] + 8, center_px[1] - 8),
                        (0, 255, 255),
                        0.5,
                        1
                    )

                if SHOW_CENTER_COORD_TEXT:
                    x_mm = center_cam[0] * 1000.0
                    y_mm = center_cam[1] * 1000.0
                    z_mm = center_cam[2] * 1000.0

                    tag_center_px = tuple(np.mean(corners[i][0], axis=0).astype(int))
                    draw_text(display, f"Cx={x_mm:.1f}mm",
                              (tag_center_px[0] + 10, tag_center_px[1] + 20),
                              (255, 255, 0), 0.5, 1)
                    draw_text(display, f"Cy={y_mm:.1f}mm",
                              (tag_center_px[0] + 10, tag_center_px[1] + 40),
                              (255, 255, 0), 0.5, 1)
                    draw_text(display, f"Cz={z_mm:.1f}mm",
                              (tag_center_px[0] + 10, tag_center_px[1] + 60),
                              (255, 255, 0), 0.5, 1)

            if len(cube_centers_cam) > 0:
                cube_centers_cam = np.array(cube_centers_cam, dtype=np.float64)
                cube_center_avg = np.mean(cube_centers_cam, axis=0)

                # 姿态只用 IMU
                R_draw = euler_to_rotation_matrix(
                    imu["roll"], imu["pitch"], imu["yaw"]
                )

                avg_center_px = tuple(
                    np.mean(np.array(center_pixels, dtype=np.float64), axis=0).astype(int)
                )

                smoothed_center_px = ema_filter(smoothed_center_px, avg_center_px, EMA_ALPHA)
                smoothed_center_px_int = tuple(np.round(smoothed_center_px).astype(int))

                if SHOW_TRAJECTORY:
                    update_trajectory(
                        trajectory,
                        smoothed_center_px_int,
                        now_ts,
                        keep_seconds=TRAJECTORY_SECONDS
                    )
                    draw_trajectory(display, trajectory, now_ts)

                if SHOW_CENTER_POINT:
                    cv2.circle(display, smoothed_center_px_int, 6, (0, 0, 255), -1)

                if SHOW_CENTER_AXES:
                    draw_center_axes_from_pose(
                        display,
                        cube_center_avg,
                        R_draw,
                        CENTER_AXIS_LENGTH,
                        camera_matrix,
                        dist_coeffs
                    )

                if SHOW_AVG_TEXT:
                    avg_text = (
                        f"Center(mm): "
                        f"X={cube_center_avg[0]*1000:.1f}, "
                        f"Y={cube_center_avg[1]*1000:.1f}, "
                        f"Z={cube_center_avg[2]*1000:.1f}"
                    )
                    draw_text(display, avg_text, (20, 30), (0, 255, 0), 0.7, 2)

                imu_text = (
                    f"IMU RPY(deg): "
                    f"R={imu['roll']:.1f} "
                    f"P={imu['pitch']:.1f} "
                    f"Y={imu['yaw']:.1f}"
                )
                draw_text(display, imu_text, (20, 60), (255, 0, 0), 0.7, 2)

                draw_text(display, f"POT={imu['pot']}  ST={imu['stationary']}",
                          (20, 90), (0, 200, 255), 0.7, 2)

                if SHOW_3D_VIEW and ax3d is not None:
                    ax3d.cla()
                    setup_3d_axes(ax3d, center=cube_center_avg)

                    # 相机原点
                    ax3d.scatter(0, 0, 0, c='k', s=40)
                    ax3d.text(0, 0, 0, "Camera")

                    # 相机坐标轴
                    ax3d.plot([0, 0.03], [0, 0], [0, 0], 'r--')
                    ax3d.plot([0, 0], [0, 0.03], [0, 0], 'g--')
                    ax3d.plot([0, 0], [0, 0], [0, 0.03], 'b--')

                    # 方块位置来自视觉，姿态来自IMU
                    draw_cube_3d(ax3d, R_draw, cube_center_avg, size=CUBE_SIZE, color='cyan', alpha=0.5)
                    draw_axes_3d(ax3d, R_draw, cube_center_avg, axis_len=CENTER_AXIS_LENGTH * 2.0)

                    ax3d.scatter(cube_center_avg[0], cube_center_avg[1], cube_center_avg[2], c='m', s=50)
                    ax3d.text(
                        cube_center_avg[0],
                        cube_center_avg[1],
                        cube_center_avg[2],
                        f"({cube_center_avg[0]:.3f},{cube_center_avg[1]:.3f},{cube_center_avg[2]:.3f})"
                    )

                    plt.draw()
                    plt.pause(0.001)

                print(
                    f"\rCenter(m): "
                    f"X={cube_center_avg[0]:.4f}, "
                    f"Y={cube_center_avg[1]:.4f}, "
                    f"Z={cube_center_avg[2]:.4f} | "
                    f"RPY=({imu['roll']:.1f},{imu['pitch']:.1f},{imu['yaw']:.1f})              ",
                    end=""
                )
            else:
                update_trajectory(trajectory, None, now_ts, keep_seconds=TRAJECTORY_SECONDS)
                if SHOW_TRAJECTORY:
                    draw_trajectory(display, trajectory, now_ts)

                draw_text(display, "No valid marker pose", (20, 30), (0, 0, 255), 0.8, 2)

                if SHOW_3D_VIEW and ax3d is not None:
                    ax3d.cla()
                    setup_3d_axes(ax3d, center=None)
                    plt.draw()
                    plt.pause(0.001)

        else:
            update_trajectory(trajectory, None, now_ts, keep_seconds=TRAJECTORY_SECONDS)
            if SHOW_TRAJECTORY:
                draw_trajectory(display, trajectory, now_ts)

            draw_text(display, "No marker detected", (20, 30), (0, 0, 255), 0.8, 2)
            draw_text(display,
                      f"IMU RPY: {imu['roll']:.1f}, {imu['pitch']:.1f}, {imu['yaw']:.1f}",
                      (20, 60), (255, 0, 0), 0.7, 2)

            if SHOW_3D_VIEW and ax3d is not None:
                ax3d.cla()
                setup_3d_axes(ax3d, center=None)
                plt.draw()
                plt.pause(0.001)

        cv2.imshow("IMU Pose + ArUco Center", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    print()
    cap.release()
    cv2.destroyAllWindows()

    if SHOW_3D_VIEW:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()