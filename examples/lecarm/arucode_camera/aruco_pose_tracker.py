import cv2
import numpy as np
import time
from collections import deque

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

# =========================================================
# 2. 显示控制
# =========================================================

SHOW_DETECTED_MARKERS = True     # 是否显示检测到的marker边框
SHOW_CENTER_POINT = True         # 是否显示平均中心点
SHOW_CENTER_AXES = True           # 是否显示cube中心坐标轴
SHOW_CENTER_ID = True            # 是否显示每个marker对应ID
SHOW_CENTER_COORD_TEXT = True    # 是否显示每个marker算出的中心坐标
SHOW_AVG_TEXT = True              # 是否显示平均中心坐标
SHOW_TRAJECTORY = True            # 是否显示过去3秒轨迹

# =========================================================
# 3. 轨迹参数
# =========================================================

TRAJECTORY_SECONDS = 3.0
TRAJECTORY_THICKNESS = 2
TRAJECTORY_CIRCLE_RADIUS = 3
TRAJECTORY_FADE = True

# 中心坐标轴长度（米）
CENTER_AXIS_LENGTH = 0.02

# 轨迹平滑系数，越接近1越平滑
EMA_ALPHA = 0.25

# =========================================================
# 4. cube / face 定义
# =========================================================
# cube 坐标系：
# +X = front
# +Y = right
# +Z = up

# 你给过的信息：
# 2 和 3 的物理贴纸位置贴反，因此这里修正
MARKER_TO_FACE = {
    0: "front",
    1: "back",
    2: "left",    # 修正
    3: "right",   # 修正
    4: "top",
}

# 每个面的贴纸平面内逆时针旋转次数（90度为1次）
DEFAULT_FACE_TURNS_CCW = {
    0: 0,
    1: 4,
    2: 4,
    3: 4,
    4: 2,
}

# =========================================================
# 5. 工具函数：标定 / 检测
# =========================================================

def load_calibration(filepath):
    """
    加载相机标定参数
    """
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
    """
    创建 ArUco 检测器
    """
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
    """
    检测图像中的ArUco标记
    """
    if detector is not None:
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=detector_params
        )
    return corners, ids, rejected

# =========================================================
# 6. 工具函数：旋转矩阵
# =========================================================

def rot_z_90_ccw():
    """
    绕 Z 轴逆时针旋转 90 度
    """
    return np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1],
    ], dtype=np.float64)


def rot_z_k_ccw(k):
    """
    绕 Z 轴逆时针旋转 k 次 90 度
    """
    k = k % 4
    R = np.eye(3, dtype=np.float64)
    for _ in range(k):
        R = rot_z_90_ccw() @ R
    return R

# =========================================================
# 7. 工具函数：face基础朝向
# =========================================================
# 定义“未做平面内旋转前”，marker坐标如何映射到cube坐标
#
# 含义：
# v_cube = R_cube_from_marker @ v_marker
#
# marker默认坐标：
# +X / +Y 在贴纸平面内
# +Z 为贴纸法向（朝外）

FACE_BASE_ROT = {
    # top: marker z -> cube +Z
    #      marker x -> cube +X
    #      marker y -> cube +Y
    "top": np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], dtype=np.float64),

    # front: marker z -> cube +X
    #        marker x -> cube +Y
    #        marker y -> cube +Z
    "front": np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=np.float64),

    # right: marker z -> cube +Y
    #        marker x -> cube -X
    #        marker y -> cube +Z
    "right": np.array([
        [0, -1, 0],
        [0,  0, 1],
        [1,  0, 0],
    ], dtype=np.float64),

    # back: marker z -> cube -X
    #       marker x -> cube -Y
    #       marker y -> cube +Z
    "back": np.array([
        [ 0,  0, -1],
        [-1,  0,  0],
        [ 0,  1,  0],
    ], dtype=np.float64),

    # left: marker z -> cube -Y
    #       marker x -> cube +X
    #       marker y -> cube +Z
    "left": np.array([
        [1,  0,  0],
        [0,  0, -1],
        [0,  1,  0],
    ], dtype=np.float64),
}

# =========================================================
# 8. 工具函数：marker -> cube 姿态
# =========================================================

def get_marker_to_cube_rotation(marker_id):
    """
    返回 R_cube_from_marker
    满足：v_cube = R_cube_from_marker @ v_marker
    """
    if marker_id not in MARKER_TO_FACE:
        raise KeyError(f"marker_id {marker_id} 未定义 face")

    face = MARKER_TO_FACE[marker_id]
    R_base = FACE_BASE_ROT[face]

    turns = DEFAULT_FACE_TURNS_CCW.get(marker_id, 0)
    R_turn = rot_z_k_ccw(turns)

    # 先做 marker 平面内旋转，再映射到 cube
    R_cube_from_marker = R_base @ R_turn
    return R_cube_from_marker


def marker_pose_to_cube_pose(rvec, tvec, cube_size, marker_id):
    """
    根据 marker 位姿，计算：
    1) cube中心在相机坐标系下的位置 center_cam
    2) cube坐标系到相机坐标系的旋转 R_cube_to_cam

    Returns:
        center_cam: shape (3,)
        R_cube_to_cam: shape (3, 3)
    """
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)

    # marker -> camera
    R_marker_to_cam, _ = cv2.Rodrigues(rvec)

    # marker -> cube
    R_cube_from_marker = get_marker_to_cube_rotation(marker_id)

    # cube -> marker
    R_marker_from_cube = R_cube_from_marker.T

    # cube -> camera
    R_cube_to_cam = R_marker_to_cam @ R_marker_from_cube

    face = MARKER_TO_FACE[marker_id]

    # face中心 -> cube中心，在 cube 坐标系中表达
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

    # 转成 marker 坐标系
    center_in_marker = R_marker_from_cube @ center_in_cube.reshape(3, 1)

    # 再转到相机坐标系
    center_cam = R_marker_to_cam @ center_in_marker + tvec

    return center_cam.reshape(3), R_cube_to_cam

# =========================================================
# 9. 工具函数：投影 / 绘图
# =========================================================

def project_cam_point(point_cam, camera_matrix, dist_coeffs):
    """
    将相机坐标系下的3D点投影到图像
    """
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


def draw_center_axes_from_pose(display, center_cam, R_cube_to_cam, axis_length,
                               camera_matrix, dist_coeffs):
    """
    在 cube 中心绘制一套坐标轴
    """
    center_cam = np.asarray(center_cam, dtype=np.float64).reshape(3)
    R_cube_to_cam = np.asarray(R_cube_to_cam, dtype=np.float64).reshape(3, 3)

    # 旋转矩阵三列分别是 cube 的 X/Y/Z 轴在相机坐标系中的方向
    x_dir = R_cube_to_cam[:, 0]
    y_dir = R_cube_to_cam[:, 1]
    z_dir = R_cube_to_cam[:, 2]

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
# 10. 工具函数：旋转平均
# =========================================================

def average_rotation_matrices(rotations):
    """
    对多个旋转矩阵做简单平均，再用SVD投回SO(3)
    """
    if len(rotations) == 0:
        return np.eye(3, dtype=np.float64)

    M = np.zeros((3, 3), dtype=np.float64)
    for R in rotations:
        M += R
    M /= len(rotations)

    U, _, Vt = np.linalg.svd(M)
    R_avg = U @ Vt

    # 保证 det(R)=+1
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt

    return R_avg

# =========================================================
# 11. 工具函数：轨迹
# =========================================================

def ema_filter(prev_pt, new_pt, alpha=0.25):
    """
    指数移动平均滤波
    """
    if prev_pt is None:
        return np.asarray(new_pt, dtype=np.float64)
    prev_pt = np.asarray(prev_pt, dtype=np.float64)
    new_pt = np.asarray(new_pt, dtype=np.float64)
    return alpha * new_pt + (1.0 - alpha) * prev_pt


def update_trajectory(trajectory, point_px, now_ts, keep_seconds=3.0):
    """
    trajectory: deque[(timestamp, (x, y))]
    """
    if point_px is not None:
        trajectory.append((now_ts, point_px))

    while trajectory and (now_ts - trajectory[0][0] > keep_seconds):
        trajectory.popleft()


def draw_trajectory(display, trajectory, now_ts):
    """
    绘制过去几秒轨迹，支持渐隐
    """
    if len(trajectory) < 1:
        return

    pts = [p for _, p in trajectory]
    ts_list = [t for t, _ in trajectory]

    for i, p in enumerate(pts):
        if TRAJECTORY_FADE:
            age = now_ts - ts_list[i]
            ratio = max(0.0, min(1.0, 1.0 - age / TRAJECTORY_SECONDS))
            # 用紫色系渐隐：越旧越暗
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
# 12. 主程序
# =========================================================

def main():
    camera_matrix, dist_coeffs = load_calibration(CALIB_FILE)
    print("[INFO] 已加载相机内参")
    print("camera_matrix =")
    print(camera_matrix)
    print("dist_coeffs =")
    print(dist_coeffs)

    detector, aruco_dict, detector_params = create_detector()

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头")
        return

    trajectory = deque()
    smoothed_center_px = None

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
        cube_rotations = []
        center_pixels = []

        now_ts = time.time()

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
                    center_cam, R_cube_to_cam = marker_pose_to_cube_pose(
                        rvec, tvec, CUBE_SIZE, marker_id
                    )
                except Exception as e:
                    print(f"[WARN] marker {marker_id} 计算失败: {e}")
                    continue

                cube_centers_cam.append(center_cam)
                cube_rotations.append(R_cube_to_cam)

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

                # 平均旋转，得到一套统一的cube姿态
                R_draw = average_rotation_matrices(cube_rotations)

                # 平均中心像素位置
                avg_center_px = tuple(
                    np.mean(np.array(center_pixels, dtype=np.float64), axis=0).astype(int)
                )

                # 平滑像素中心
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
                        f"Cube Center Avg (mm): "
                        f"X={cube_center_avg[0]*1000:.1f}, "
                        f"Y={cube_center_avg[1]*1000:.1f}, "
                        f"Z={cube_center_avg[2]*1000:.1f}"
                    )
                    draw_text(display, avg_text, (20, 30), (0, 255, 0), 0.7, 2)

                print(
                    f"\rCube Center Avg (m): "
                    f"X={cube_center_avg[0]:.4f}, "
                    f"Y={cube_center_avg[1]:.4f}, "
                    f"Z={cube_center_avg[2]:.4f}   ",
                    end=""
                )
            else:
                update_trajectory(trajectory, None, now_ts, keep_seconds=TRAJECTORY_SECONDS)
                if SHOW_TRAJECTORY:
                    draw_trajectory(display, trajectory, now_ts)
                draw_text(display, "No valid marker pose", (20, 30), (0, 0, 255), 0.8, 2)

        else:
            update_trajectory(trajectory, None, now_ts, keep_seconds=TRAJECTORY_SECONDS)
            if SHOW_TRAJECTORY:
                draw_trajectory(display, trajectory, now_ts)
            draw_text(display, "No marker detected", (20, 30), (0, 0, 255), 0.8, 2)

        cv2.imshow("Aruco Cube Center", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    print()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()