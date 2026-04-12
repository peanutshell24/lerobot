import cv2
import glob
import os

# ========= 1. 基础配置 =========
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

detector_params = cv2.aruco.DetectorParameters()
detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

square_length = 0.028
marker_length = 0.021

image_files = glob.glob("charuco_images/*.jpg")
print(f"找到 {len(image_files)} 张图片")

if len(image_files) == 0:
    raise RuntimeError("charuco_images/ 目录下没有 jpg 图片")

# ========= 2. 候选 board 配置 =========
# 重点测试：
# - 9x6
# - 6x9
# - 是否启用 legacy pattern
candidate_boards = [
    ("9x6_normal", 9, 6, False),
    ("9x6_legacy", 9, 6, True),
    ("6x9_normal", 6, 9, False),
    ("6x9_legacy", 6, 9, True),
]

def make_board(sx, sy, legacy=False):
    board = cv2.aruco.CharucoBoard(
        (sx, sy),
        square_length,
        marker_length,
        aruco_dict
    )
    if legacy and hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(True)
    return board

# ========= 3. 先测试哪种 board 真能插值出角点 =========
best_name = None
best_board = None
best_score = -1

for name, sx, sy, legacy in candidate_boards:
    board = make_board(sx, sy, legacy)
    total_charuco = 0
    ok_images = 0

    print(f"\n===== 测试配置: {name} =====")

    for fname in image_files:
        img = cv2.imread(fname)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=detector_params
        )

        n_markers = 0 if ids is None else len(ids)

        if ids is None or len(ids) == 0:
            print(f"{fname}: markers=0, charuco=0")
            continue

        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=board
        )

        n_charuco = 0 if ret is None else int(ret)
        print(f"{fname}: markers={n_markers}, charuco={n_charuco}")

        total_charuco += n_charuco
        if n_charuco >= 4:
            ok_images += 1

    print(f"配置 {name} 总角点数: {total_charuco}, 有效图片数: {ok_images}")

    if total_charuco > best_score:
        best_score = total_charuco
        best_name = name
        best_board = board

print("\n==============================")
print(f"最佳配置: {best_name}, score={best_score}")
print("==============================")

if best_score <= 0 or best_board is None:
    raise RuntimeError(
        "所有候选 board 配置都无法插值出 Charuco 角点。"
        "这通常说明：\n"
        "1) 你代码里的 board 行列数和实际打印板不一致；\n"
        "2) 这批图片拍的是另一张板；\n"
        "3) 字典不是 DICT_4X4_50。"
    )

# ========= 4. 用最佳 board 正式标定 =========
all_charuco_corners = []
all_charuco_ids = []
image_size = None

os.makedirs("debug_vis", exist_ok=True)

for fname in image_files:
    img = cv2.imread(fname)
    if img is None:
        print(f"[跳过] 读图失败: {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]

    corners, ids, rejected = cv2.aruco.detectMarkers(
        gray, aruco_dict, parameters=detector_params
    )

    vis = img.copy()

    if ids is None or len(ids) == 0:
        cv2.imwrite(f"debug_vis/{os.path.basename(fname)}", vis)
        continue

    ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        markerCorners=corners,
        markerIds=ids,
        image=gray,
        board=best_board
    )

    cv2.aruco.drawDetectedMarkers(vis, corners, ids)

    n_charuco = 0 if ret is None else int(ret)
    print(f"{fname}: charuco={n_charuco}")

    if ret is not None and ret >= 8:
        cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)
        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)

    cv2.imwrite(f"debug_vis/{os.path.basename(fname)}", vis)

if len(all_charuco_corners) < 5:
    raise RuntimeError(f"可用标定图太少（有效图 {len(all_charuco_corners)} < 5）")

retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    charucoCorners=all_charuco_corners,
    charucoIds=all_charuco_ids,
    board=best_board,
    imageSize=image_size,
    cameraMatrix=None,
    distCoeffs=None
)

print("\n===== 标定结果 =====")
print("使用 board 配置:", best_name)
print("重投影误差:", retval)
print("camera_matrix =\n", camera_matrix)
print("dist_coeffs =\n", dist_coeffs)

fs = cv2.FileStorage("camera_charuco.yml", cv2.FILE_STORAGE_WRITE)
fs.write("camera_matrix", camera_matrix)
fs.write("dist_coeffs", dist_coeffs)
fs.release()

print("\n已保存到 camera_charuco.yml")
print("调试图在 debug_vis/")