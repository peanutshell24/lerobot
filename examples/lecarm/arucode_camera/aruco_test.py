import cv2
import numpy as np
import time
from collections import deque
import matplotlib.pyplot as plt

# =========================================================
# 1. 配置参数
# =========================================================

MARKER_LENGTH = 0.028   # ArUco边长（米）
CUBE_SIZE = 0.05        # 立方体边长（米）

ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_50
CALIB_FILE = "camera_charuco.yml"
CAMERA_ID = 0

ENABLE_TEMPORAL_SMOOTHING = True
POSE_EMA_ALPHA = 0.25

SHOW_DETECTED_MARKERS = True
SHOW_BOARD_AXES = True
SHOW_TEXT = True
SHOW_TRAJECTORY = True

AXIS_LENGTH = 0.03
TRAJECTORY_SECONDS = 3.0
TRAJECTORY_THICKNESS = 2
TRAJECTORY_CIRCLE_RADIUS = 3
TRAJECTORY_FADE = True

# 3D窗口
SHOW_3D_WINDOW = True
MAX_3D_POINTS = 300
WORLD_AXIS_LIMIT = 0.30   # 3D坐标轴显示范围，单位米

# =========================================================
# 2. cube / face 定义
# =========================================================
# cube 坐标系：
# +X = front
# +Y = right
# +Z = up

MARKER_TO_FACE = {
    0: "front",
    1: "back",
    2: "left",
    3: "right",
    4: "top",
}

DEFAULT_FACE_TURNS_CCW = {
    0: 1,
    1: 3,
    2: 3,
    3: 1,
    4: 2,
}

# =========================================================
# 3. 工具函数：标定 / 检测器
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
    else:
        detector = None

    return detector, aruco_dict, detector_params


def detect_markers(gray, detector, aruco_dict, detector_params):
    if detector is not None:
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=detector_params
        )
    return corners, ids, rejected

# =========================================================
# 4. 旋转工具
# =========================================================

def rot_z_k_ccw(k):
    k = k % 4
    theta = k * (np.pi / 2.0)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ], dtype=np.float64)


def average_two_rotations(R_prev, R_new, alpha):
    M = (1.0 - alpha) * R_prev + alpha * R_new
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def ema_vec(prev_vec, new_vec, alpha):
    if prev_vec is None:
        return new_vec.copy()
    return (1.0 - alpha) * prev_vec + alpha * new_vec

# =========================================================
# 5. face 基础朝向
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
# 6. 构造 board
# =========================================================

def get_face_center_in_cube(face, cube_size):
    h = cube_size / 2.0
    if face == "front":
        return np.array([ h, 0.0, 0.0], dtype=np.float64)
    elif face == "back":
        return np.array([-h, 0.0, 0.0], dtype=np.float64)
    elif face == "right":
        return np.array([0.0,  h, 0.0], dtype=np.float64)
    elif face == "left":
        return np.array([0.0, -h, 0.0], dtype=np.float64)
    elif face == "top":
        return np.array([0.0, 0.0,  h], dtype=np.float64)
    else:
        raise ValueError(f"未知 face: {face}")


def get_marker_corners_in_cube(marker_id, marker_length, cube_size):
    if marker_id not in MARKER_TO_FACE:
        raise KeyError(f"marker_id {marker_id} 未定义 face")

    face = MARKER_TO_FACE[marker_id]
    R_base = FACE_BASE_ROT[face]
    turns = DEFAULT_FACE_TURNS_CCW.get(marker_id, 0)
    R_turn = rot_z_k_ccw(turns)

    R_cube_from_marker = R_base @ R_turn

    center_cube = get_face_center_in_cube(face, cube_size).reshape(3, 1)
    s = marker_length / 2.0

    local_corners = [
        np.array([-s,  s, 0.0], dtype=np.float64).reshape(3, 1),
        np.array([ s,  s, 0.0], dtype=np.float64).reshape(3, 1),
        np.array([ s, -s, 0.0], dtype=np.float64).reshape(3, 1),
        np.array([-s, -s, 0.0], dtype=np.float64).reshape(3, 1),
    ]

    corners_cube = []
    for p_local in local_corners:
        p_cube = center_cube + R_cube_from_marker @ p_local
        corners_cube.append(p_cube.reshape(3).astype(np.float32))

    return np.array(corners_cube, dtype=np.float32)


def create_cube_board(aruco_dict, marker_length, cube_size):
    board_ids = []
    board_corners = []

    for marker_id in sorted(MARKER_TO_FACE.keys()):
        corners = get_marker_corners_in_cube(marker_id, marker_length, cube_size)
        board_ids.append([marker_id])
        board_corners.append(corners)

    board_ids = np.array(board_ids, dtype=np.int32)

    if hasattr(cv2.aruco, "Board"):
        board = cv2.aruco.Board(board_corners, aruco_dict, board_ids)
    else:
        board = cv2.aruco.Board_create(board_corners, aruco_dict, board_ids)

    return board

# =========================================================
# 7. 投影 / 绘图
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


def update_trajectory(trajectory, point_px, now_ts, keep_seconds=3.0):
    if point_px is not None:
        trajectory.append((now_ts, point_px))
    while trajectory and (now_ts - trajectory[0][0] > keep_seconds):
        trajectory.popleft()

# =========================================================
# 8. pose 平滑
# =========================================================

class PoseSmoother:
    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self.tvec = None
        self.R = None
        self.initialized = False

    def update(self, rvec, tvec):
        R_new, _ = cv2.Rodrigues(rvec)
        t_new = np.asarray(tvec, dtype=np.float64).reshape(3, 1)

        if not self.initialized:
            self.R = R_new.copy()
            self.tvec = t_new.copy()
            self.initialized = True
        else:
            self.R = average_two_rotations(self.R, R_new, self.alpha)
            self.tvec = ema_vec(self.tvec, t_new, self.alpha)

        rvec_smooth, _ = cv2.Rodrigues(self.R)
        return rvec_smooth.reshape(3, 1), self.tvec.reshape(3, 1)

# =========================================================
# 9. 3D 可视化窗口
# =========================================================

class Realtime3DViewer:
    def __init__(self, max_points=300, axis_limit=0.3):
        self.max_points = max_points
        self.axis_limit = axis_limit
        self.points = deque(maxlen=max_points)

        plt.ion()
        self.fig = plt.figure("Cube 3D Trajectory")
        self.ax = self.fig.add_subplot(111, projection='3d')

    def update(self, point):
        if point is not None:
            self.points.append(np.asarray(point, dtype=np.float64).reshape(3))
        self.redraw()

    def redraw(self):
        self.ax.cla()

        self.ax.set_title("Cube Center in Camera 3D Space")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")

        lim = self.axis_limit
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
        self.ax.set_zlim(0.0, lim * 2.0)

        # 相机原点
        self.ax.scatter([0], [0], [0], s=60, marker='o')
        self.ax.text(0, 0, 0, "Camera")

        # 相机坐标轴
        axis_len = lim * 0.35
        self.ax.plot([0, axis_len], [0, 0], [0, 0])  # X
        self.ax.plot([0, 0], [0, axis_len], [0, 0])  # Y
        self.ax.plot([0, 0], [0, 0], [0, axis_len])  # Z
        self.ax.text(axis_len, 0, 0, "X")
        self.ax.text(0, axis_len, 0, "Y")
        self.ax.text(0, 0, axis_len, "Z")

        if len(self.points) > 0:
            pts = np.array(self.points)
            xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]

            self.ax.plot(xs, ys, zs, linewidth=2)
            self.ax.scatter(xs[-1:], ys[-1:], zs[-1:], s=50, marker='o')
            self.ax.text(xs[-1], ys[-1], zs[-1],
                         f"({xs[-1]:.3f}, {ys[-1]:.3f}, {zs[-1]:.3f})")

        self.ax.view_init(elev=25, azim=-60)
        plt.draw()
        plt.pause(0.001)

    def close(self):
        plt.ioff()
        plt.close(self.fig)

# =========================================================
# 10. 主程序
# =========================================================

def main():
    camera_matrix, dist_coeffs = load_calibration(CALIB_FILE)
    detector, aruco_dict, detector_params = create_detector()
    board = create_cube_board(aruco_dict, MARKER_LENGTH, CUBE_SIZE)

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头")
        return

    print("[INFO] 已加载相机参数")
    print("camera_matrix =")
    print(camera_matrix)
    print("dist_coeffs =")
    print(dist_coeffs)

    print("[INFO] 已创建 cube board")
    print("按 q 或 ESC 退出")

    pose_smoother = PoseSmoother(alpha=POSE_EMA_ALPHA)
    trajectory = deque()
    viewer3d = Realtime3DViewer(MAX_3D_POINTS, WORLD_AXIS_LIMIT) if SHOW_3D_WINDOW else None

    prev_rvec = None
    prev_tvec = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] 读取摄像头失败")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display = frame.copy()
        now_ts = time.time()

        corners, ids, _ = detect_markers(gray, detector, aruco_dict, detector_params)

        if ids is not None and len(ids) > 0:
            valid_corners = []
            valid_ids = []
            for i in range(len(ids)):
                marker_id = int(ids[i][0])
                if marker_id in MARKER_TO_FACE:
                    valid_corners.append(corners[i])
                    valid_ids.append([marker_id])

            if len(valid_ids) > 0:
                valid_ids = np.array(valid_ids, dtype=np.int32)

                if SHOW_DETECTED_MARKERS:
                    cv2.aruco.drawDetectedMarkers(display, valid_corners, valid_ids)

                try:
                    if prev_rvec is None or prev_tvec is None:
                        retval, rvec, tvec = cv2.aruco.estimatePoseBoard(
                            valid_corners, valid_ids, board,
                            camera_matrix, dist_coeffs, None, None
                        )
                    else:
                        try:
                            retval, rvec, tvec = cv2.aruco.estimatePoseBoard(
                                valid_corners, valid_ids, board,
                                camera_matrix, dist_coeffs,
                                prev_rvec, prev_tvec, True
                            )
                        except TypeError:
                            retval, rvec, tvec = cv2.aruco.estimatePoseBoard(
                                valid_corners, valid_ids, board,
                                camera_matrix, dist_coeffs,
                                prev_rvec, prev_tvec
                            )
                except Exception as e:
                    retval = 0
                    print(f"[WARN] estimatePoseBoard 失败: {e}")

                if retval > 0:
                    prev_rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
                    prev_tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)

                    if ENABLE_TEMPORAL_SMOOTHING:
                        rvec_draw, tvec_draw = pose_smoother.update(prev_rvec, prev_tvec)
                    else:
                        rvec_draw, tvec_draw = prev_rvec, prev_tvec

                    if SHOW_BOARD_AXES:
                        if hasattr(cv2, "drawFrameAxes"):
                            cv2.drawFrameAxes(
                                display,
                                camera_matrix,
                                dist_coeffs,
                                rvec_draw,
                                tvec_draw,
                                AXIS_LENGTH
                            )
                        else:
                            cv2.aruco.drawAxis(
                                display,
                                camera_matrix,
                                dist_coeffs,
                                rvec_draw,
                                tvec_draw,
                                AXIS_LENGTH
                            )

                    center_cam = np.asarray(tvec_draw, dtype=np.float64).reshape(3)
                    center_px = project_cam_point(center_cam, camera_matrix, dist_coeffs)
                    update_trajectory(trajectory, center_px, now_ts, TRAJECTORY_SECONDS)

                    if SHOW_TRAJECTORY:
                        draw_trajectory(display, trajectory, now_ts)

                    cv2.circle(display, center_px, 5, (0, 0, 255), -1)

                    if SHOW_TEXT:
                        draw_text(
                            display,
                            f"Cube Center (m): X={center_cam[0]:.4f}  Y={center_cam[1]:.4f}  Z={center_cam[2]:.4f}",
                            (20, 35),
                            (0, 255, 0),
                            0.65,
                            2
                        )

                    if viewer3d is not None:
                        viewer3d.update(center_cam)

                    print(
                        f"\rCube Center (m): X={center_cam[0]:.4f}, "
                        f"Y={center_cam[1]:.4f}, Z={center_cam[2]:.4f}   ",
                        end=""
                    )
                else:
                    update_trajectory(trajectory, None, now_ts, TRAJECTORY_SECONDS)
                    if SHOW_TRAJECTORY:
                        draw_trajectory(display, trajectory, now_ts)
                    draw_text(display, "Board pose solve failed", (20, 30), (0, 0, 255), 0.8, 2)

                    if viewer3d is not None:
                        viewer3d.update(None)
            else:
                update_trajectory(trajectory, None, now_ts, TRAJECTORY_SECONDS)
                if SHOW_TRAJECTORY:
                    draw_trajectory(display, trajectory, now_ts)
                draw_text(display, "No configured marker detected", (20, 30), (0, 0, 255), 0.8, 2)

                if viewer3d is not None:
                    viewer3d.update(None)
        else:
            update_trajectory(trajectory, None, now_ts, TRAJECTORY_SECONDS)
            if SHOW_TRAJECTORY:
                draw_trajectory(display, trajectory, now_ts)
            draw_text(display, "No marker detected", (20, 30), (0, 0, 255), 0.8, 2)

            if viewer3d is not None:
                viewer3d.update(None)

        cv2.imshow("Aruco Cube Board Pose", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    print()
    cap.release()
    cv2.destroyAllWindows()
    if viewer3d is not None:
        viewer3d.close()


if __name__ == "__main__":
    main()